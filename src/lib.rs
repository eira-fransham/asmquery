//! This is a sketch of how a vertical slice through the x86 assembly pipeline might look
//! like.
//!
//! Here's a quick overview/glossary:
//! - Unless otherwise specified, I'll use "instruction" to refer to the actual instructions
//!   on the machine. In the machine spec, multiple instruction definitons could correspond
//!   to a single opcode, as different sets of arguments are classed as different instruction
//!   definitions. Although register-register add and register-immediate add are one
//!   "instruction", they are two separate instruction _definitions_, although almost
//!   certainly we will have abstractions that allow defining a single operation with all of
//!   the forms for register-register, register-immediate, register-memory and so on auto-
//!   generated.
//!
//! - The biggest difference between this and other similar libraries is that the machine
//!   specification is written in terms of operations, which may have a many-to-many
//!   correspondance with actual instructions. Operations are the smallest indivisible piece
//!   of an instruction. An `add` is an operation, but an "`add` and set the carry flag if
//!   it overflows" is two separate operations, as there exist instructions that can add
//!   without setting the carry flag.
//!
//! - To compile a Wasm operation to assembly, we write a low-level infinite register machine
//!   IR that I'm going to call Low IR (just so that we have some name to refer to it that
//!   won't be confused with the actual instructions on the machine). This IR is defined in
//!   terms of _operations_, which take some number of inputs and produce precisely one
//!   output. On most machines, many instructions produce multiple outputs when executed. For
//!   example, the `add` instruction might set the carry flag in the `FLAGS` register. We
//!   would respresent this as two independent Low IR instructions. With an LLVM IR-style
//!   strawman syntax, that might look like this:
//!   ```
//!   %2 = add %0, %1
//!   %3 = add_carry %0, %1
//!   ```
//!   `add_carry` is _not_ add-with-carry, it simply asks whether the add operation with
//!   the provided operands would set the carry bit. On x86, one instruction will do both of
//!   these operations simultaneously, and so we combine them into one. The algorithm that
//!   does this is provided below.
//!
//! - I'll refer to the machine-specific code that defines the Low IR for each Lightbeam IR
//!   instruction as the "Low IR generator", since for our purposes here the only important
//!   thing is that it generates Low IR. The fact that it generates it from Lightbeam IR,
//!   which in turn is generated from WebAssembly, is not particularly important for our
//!   purposes here. Since a large number of operations are valid on every machine - every
//!   machine will have an add, a subtract, a load, a store and so forth - most of this
//!   code can be provided using a default implementation, with relatively few instructions
//!   needing machine-specific implementations.
//!
//! - I'm going to refer to the code that compiles Low IR into actual instructions on the
//!   machine as the Low IR compiler, or LIRC. This code handles allocating real locations
//!   in registers or on the stack for the virtual registers defined in Low IR, and it
//!   handles instruction selection. It's possible for a single instruction to be selected
//!   for many Low IR operations, but impossible for more than one instruction to be
//!   selected for a single Low IR operation. It should also handle control flow, but
//!   precisely how this works is currently unclear. Since the machine specification
//!   provides lots of metadata about the machine, LIRC can be machine-independent,
//!   relying on the Low IR generator to provide Low IR valid for the machine and the
//!   machine specification to provide any metadata needed to do correct register
//!   allocation, instruction selection and control flow.
//!
//! - When I refer to "doing work at compile-time", I mean having that work done while
//!   compiling _Lightbeam_, so that work is already done by the time that Lightbeam
//!   comes to compile some WebAssembly code. I will not use "compile-time" to refer
//!   to Lightbeam compiling WebAssembly, I will say "at runtime" (i.e. when Lightbeam is
//!   running). For the runtime of the code emitted by Lightbeam, I will say words to the
//!   effect of "when we run Lightbeam's output" or "when running the outputted code".
//!
//!
//! The highest-level pseudocode looks like this:
//! * Generate a stateful machine-generic codegen backend `B`, parameterised
//!   with a stateless machine-specific Low IR instruction generator `G`
//! * Generate a stateless machine-specific assembler `A`
//! * For each WebAssembly instruction (call each instruction `WI`)
//!   * Convert `WI` to a stream of Microwasm instructions `MI`
//!   * Convert `MI` into a stream of Machine IR instructions `IRS`
//!     * Each individual Microwasm instruction in `MI` is converted into a stream of
//!       Low IR instructions by `G` and these streams are concatenated together
//!   * Create a stateful cursor `CD` into `IRS`
//!   * While `CD` has not reached the end of `IRS`
//!     * Get the current Low IR instruction `IR` pointed to by `CD` and advance the
//!       cursor
//!     * `B` fills in the virtual registers in `IR` with constraints based on its
//!       internal state to make a query `Q`
//!     * `B` passes `Q` into the assembler to get a list of matches `M`
//!     * `B` checks that there exists some match in `M` that could be emitted (making
//!       sure that e.g. data flow and clobbers are correct), and if not it returns an
//!       error.
//!     * Loop
//!       * Get the current Low IR instruction `IR'` pointed to by `CD` and advance the
//!         cursor
//!       * `B` fills in the virtual registers in `IR'` with constraints based on its
//!         internal state to make a query `Q'`
//!       * Refine `M` to only contain the matches that _also_ match `Q'` to create a new
//!         list of matches `M'`
//!       * Exit the loop if `M'` is empty or if there are no matches in `M'` that could be
//!         emitted (w.r.t. data flow, clobbers etc)
//!       * Otherwise, set `M` to be `M'` and repeat the loop
//!     * Get the best match in `M` (by some definition of "best", perhaps by which match
//!       requires the least spilling or even by cycle count)
//!     * `B` fills this match in with specific locations, so precisely one memory location,
//!       register etc., and passes this to the assembler to encode it into machine code and
//!       write it to a byte buffer
//!
//! > NOTE: We can take advantage here of the fact that the list of candidate matches is far
//! >       larger than the list of matches that we could actually emit. In fact, the candidate
//! >       matches that we get from the machine specification are deterministically and
//! >       statelessly generated from each Low IR instruction. This means that we can
//! >       memoise the query result for each Low IR instruction and store it as a bitfield,
//! >       where the "refined" matches that are candidate instructions that may be able to
//! >       produce both outputs can simply be calculated by doing the bitwise `and` of these
//! >       two bitfields. We then iterate through the remaining matches, doing the calculation
//! >       that actually needs to be done at runtime - e.g. data flow and clobbers.
//!
//! The most important thing to note here is that Low IR instrs can only be collapsed so long as
//! there exists some instruction that could collapse each intermediate step. This is because if we
//! read the next instruction and it can't be collapsed into our current state, we can't rewind to
//! find the most-recent set of instructions that _can_ be collapsed - we need to emit something and
//! continue or fail. For example, the complex operations included in x86's memory addressing modes
//! can be collapsed together since at every step we could always just emit a `lea` to do some
//! component of an addressing calculation without doing the load.
//!
//! A cool thing about this algorithm: assuming that it can be efficiently implemented this
//! gives us optimisations like converting add-and-test into add-and-use-side-effect-flags
//! while _also_ converting add-without-test into `lea` where appropriate, without explicitly
//! implementing this as an optimisation or even needing to do any tracking in Lightbeam of
//! what outputs have been produced as a side-effect and which outputs have been clobbered -
//! it's just implicit in the querying algorithm. More optimisations could be implemented
//! with better tracking of clobbering in the generic backend - such as `add, mov, test`
//! being converted into `add, mov` with the test implicit in the `add` but not clobbered by
//! the `mov` - but it's great that we get some optimisations implemented for free. This
//! algorithm will also allow the following:
//!
//! ```
//! %2 = add %0, %1
//! %3 = sub %1, %2
//! %4 = add_carry %0, %1
//! ```
//!
//! Since the `add_carry` output can't be combined with the `sub` output, a query will be performed
//! for `add_carry` alone, which will generate a new add, discard the actual value and keep only
//! the carry bit. Obviously we should avoid patterns like this where possible, but it means that
//! if we have an `add` followed by an `add_carry` but DCE eliminates the `add`, we still generate
//! correct code.
//!
//! The lowest level is the actual machine specification, which allows for both generic
//! outputs that we should expect every machine to have and the machine-specific outputs
//! that implement behaviour that is unique to each machine. This allows Lightbeam to
//! provide default implementations of many Wasm instructions to most backends using the
//! generic outputs, while still making the Rust compiler complain for any Wasm
//! instructions that can't be implemented using outputs that are in every machine. At
//! some point we might find that the concept of a generic instruction for every machine
//! we support is unhelpful and that we want to have some concept of outputs that are
//! generic across some subset of machines, but for now it's far simpler to have most
//! outputs exist for all machines with each machine having some set of unique
//! instructions for special cases.
//!
//! A machine's complex memory addressing modes can be implemented by expanding
//! the complex series of operations done as part of the memory operation into
//! a series of RISC inputs and outputs. For example, you could define x86's
//! 32-bit add with memory operand by specifying exactly how the memory operand
//! is calculated, splitting it into its component calculations, the load that it
//! performs, and the resulting addition.
//!
//! ```
//! [
//!     // LHS + load(BASE + (INDEX << SCALE) + DISP)
//!     G::Add32.output(int_reg_64, [1, 5]),
//!     // load(BASE + (INDEX << SCALE) + DISP)
//!     G::Load32.output(INTERNAL, [2]),
//!     // BASE + (INDEX << SCALE) + DISP
//!     G::Add32.output(INTERNAL, [3, 4]),
//!     // (INDEX << SCALE) + DISP
//!     G::Add32.output(INTERNAL, [4, 6]),
//!     // INDEX << SCALE
//!     G::ShiftL32.output(INTERNAL, [8, 9]),
//!     input(int_reg_32).eq(0) // LHS operand
//!     input(imm32),           // DISP
//!     input(int_reg_32),      // BASE
//!     input(int_reg_32),      // INDEX
//!     input(imm3),            // SCALE
//! ]
//! ```
//!
//! `INTERNAL` is used as the destination of these intermediate outputs. Precisely
//! how `INTERNAL` is represented isn't important, the important thing is that
//! whatever constraints it defines cannot be fulfilled. This ensures that
//! instructions that do memory operations are considered as candidates to be
//! merged together, but that the merged instruction cannot be emitted if LIRC
//! needs to allocate an actual location for any of these intermediate values,
//! for example if the intermediate result is used later on. Here's an example of
//! what some Low IR with complex memory operations would look like. This code
//! would compile to just a single instruction using the algorithm above and the
//! instruction definition that I just proposed (assume `%base`, `%index` and
//! `%lhs` were defined previously):
//!
//! ```
//! %mem0 = add %base, %disp
//! %shifted_index = shl %index, 2imm3
//! %mem1 = add %mem0, %shifted_index
//! %loaded = load %mem1
//! %added = add %lhs, %loaded
//! ```
//!
//! The reason that I think that this is a better solution to having some form of
//! explicit memory calculation is that it's a common pattern in generated Wasm code
//! to do some simple calculations followed by a memory access. This is because Wasm
//! only has pretty simple addressing modes. The compiler generating the Wasm code
//! can generally assume that this pattern can be detected and converted into x86
//! addressing instructions. If we split our memory addressing up like this, we
//! can essentially detect and coalesce this pattern for free, whereas if we try to
//! detect and generate it by inspecting the Wasm instructions and generating some
//! form of special memory calculation then we have to thread far more information
//! through the whole of the program. This gives us it for free and keeps most of
//! our code self-contained and stateless.
//!
//! It might be useful to maintain a bitmask for some subset of outputs that represents
//! the set of instructions that can actually produce that value into a specific
//! register or memory location. For example, `add reg, reg` produces an `Add32` output
//! into location that can be later accessed, whereas after emitting
//! `mov r2, [r0 + r1]` you cannot access `r0 + r1`. Since we need to check whether we
//! can emit a given instruction at every step, very quickly masking out all the
//! instructions that only have an output like `Add32` as an intermediate step would
//! make code a lot faster, especially since x86 has many, many instructions that can
//! take memory operands and otherwise we'd need to iterate through every one of them
//! just to realise that the bounds cannot be fulfilled.
//!
//! A quick note: everywhere where we use `Vec` we'd ideally use some trickery to do
//! everything on the stack and avoid allocation. Every allocation means work that
//! cannot be done at compile-time and increased difficulty figuring out complexity. I
//! have ideas of precisely how to constrain ourselves to the stack everywhere that we
//! need to be, but to keep this sample code simple I've used `Vec` for now.

/// An argument constraint, this means that this input or output must be in a specific place.
// TODO: We might want this to have some way to specify that the output needs to be a special
//       register - `rip`, `FLAGS` etc. - or a specific section of memory (since x86 has
//       different instructions depending on how big your memory operand is). The only thing
//       it shouldn't care about is concerns that span more than one query, such as data flow
struct Constraint {
    reg: bool,
    mem: bool,
    imm: bool,
}

/// Any value that can be obtained from some number of other values. This is an "output" and
/// not an "instruction", as there is a many-to-many relationship between outputs and
/// instructions - instructions can have many outputs, and it could be possible to obtain a
/// given output through one of many different instructions.
enum Output<S> {
    Generic(outputs::Generic),
    Specific(S),
}

// TODO: Handle instructions that don't return a value, like `jmp`, `return`, `ud2` etc.
//       Probably this can be handled like other outputs, but where the output is of
//       the bottom type (i.e. a dummy invalid type) and so matches all constraints.
// TODO: This doesn't give any way to avoid clobbers in the query itself, you have to
//       iterate through results. Does it matter?
struct Query<S>(Output<S>, Constraint, Vec<Constraint>);

/// A structure that represents a list of candidate instructions that can be refined by
/// applying further queries. This should be a bitfield of some kind.
struct Matches<S> {
    /* ..TODO.. */
}

impl<S> Matches<S> {
    // Get the intersection of two sets of matches
    fn merge(self, other: Self) -> Self {
        unimplemented!()
    }

    // Iterate through the indices of matches
    // TODO: Should we have `Matches` be able to directly return an iterator of instruction
    //       definitions instead of requiring Lightbeam to index into the machine manually?
    fn iter(&self) -> impl Iterator<Item = usize> {
        unimplemented!()
    }
}

/// A machine specification
trait Machine {
    /// Type of machine-specific outputs
    type SpecificOutput;

    /// Machines should be stateless, so we use `&` here, but it's possible that
    /// we might want to use `&mut` in the future.
    // This should just return a bitfield with all bits set.
    // TODO: This should be a const fn, right now Rust has no way to have
    //       `const fn`s in traits but there are ways around that.
    fn query(&self, query: Query<Self::SpecificOutput>) -> Matches<S>;
}

// I've chosen to write an embedded DSL here, so that we don't have to write
// blit-a-string-to-a-file-style codegen, but we could totally have this be an
// external DSL like in GCC instead. Writing the codegen would be more of a
// hassle and make compile-times worse though, and I think that an eDSL like
// this would work better for our usecase.
//
// If we carefully use `const fn` and the type system, this should compile to
// code as good as if we had hand-rolled it or written codegen.
//
// TODO: How do we handle control flow? The assembler should only expose very
//       simple functions around this and leave stuff like calling conventions
//       to Lightbeam, but how do we represent stuff like `je` vs `jle` and
//       so forth in a way that compiles to efficient code on x86, ARM, etc.
//
// TODO: Lightbeam currently makes heavy use of `push`/`pop`, but not all
//       architectures have this. We want to use these instructions on x86
//       since they're _far_ faster than manually reserving stack space, but
//       we ideally want other architectures without these instructions to
//       be able to reserve stack space in steps (e.g. powers of two).
//       Maybe this can just be a flag in the machine specification? Another
//       possibility is having a `stack_space_increment` function in the
//       `LightbeamBackend` trait that returns the step by which to increment
//       the amount of space reserved on the stack. The x64 machine spec
//       could then simply define `push` as an instruction with 2 outputs -
//       the memory location at `[rsp]`, where the input is moved, and `rsp`
//       itself, which is incremented by 8. Lightbeam just needs to request the
//       value to be moved and the stack pointer to be incremented by 8 and
//       the query refinement algorithm will take care of the rest. We could
//       have the same code to write to stack and increment by some amount for
//       both ARM64 and x64 and have the query refinement algorithm handle
//       converting this to a `mov` + `sub` on the former and a `push` on the
//       latter.
fn make_x64_specification() -> impl Machine {
    use outputs::Generic as G;

    fn r32(reg: Reg) -> Reg {
        use std::u32;

        reg.mask(u32::MAX)
    }

    fn r16(reg: Reg) -> Reg {
        use std::u16;

        reg.mask(u16::MAX)
    }

    fn r8(reg: Reg) -> Reg {
        use std::u8;

        reg.mask(u8::MAX)
    }

    let int_reg_64 = RegClass::new([R0, R1, R2 /* ..snip.. */]);
    let int_reg_32 = RegClass::new([r32(R0), r32(R1), r32(R2) /* ..snip.. */]);
    let float_reg_64 = RegClass::new([/* ..snip.. */]);
    let m64 = Memory.mask(64);
    let m32 = Memory.mask(32);
    let imm32 = Immediate.mask(32);

    // TODO: How to handle some registers overlapping with others? For example,
    //       some 32-bit instrs clobber the whole 64-bit reg, others only clobber
    //       part. Maybe we just don't handle this at all? I don't think we need
    //       to.
    //
    // This separation of the place that we store, say, the carry flag (i.e. FLAGS & 0x1)
    // with the actual `G::Carry` output that it represents means that we can handle
    // instructions that use the same bit to mean different things
    //
    // `anon!` is an external macro to create an anonymous struct and is just for convenience
    // here, basically acting like a namespace so we don't have to repeat `flags_cf`, `flags_pf`
    // etc - they're all in one place.
    let flags = anon! {
        cf: FLAGS.mask(0x1),
        pf: FLAGS.mask(0x4),
        af: FLAGS.mask(0x10),
        zf: FLAGS.mask(0x40),
        sf: FLAGS.mask(0x80),
        // ..snip..
        of: FLAGS.mask(0x800),
    };

    MachineSpec::new()
        .instr(
            // This is the metadata section. This specifies everything that the query engine needs,
            // so inputs, outputs, clobbers, commutativity, maybe even identity.
            [
                // TODO: We might want more granular specification of inputs than just indices,
                //       since an instruction might simultaneously use some bits of an input in
                //       one output and other bits in another output. For example, if we want
                //       the parity of the full 32-bit add output we will need to use a dedicated
                //       instruction or series of instructions, but if we only need the parity of
                //       the LSB then we can reuse the parity result from this `Add32`. Currently I
                //       don't think this is necessary though, so I've just used indices with
                //       no other information. It will probably look similar to the `.mask` syntax
                //       that I have above.
                //
                // Calling `.output` on value specifying an output type (for example, `outputs::Generic`)
                // creates a marker that this instruction can create this output for the given inputs
                // into some register in the class `int_reg_64`.
                //
                // Even though this is a 32-bit add I've used a 64-bit register, since a 32-bit add
                // will zero the upper bits of the register. Not sure if we could have a better way
                // of specifying this.
                G::Add32.output(int_reg_64, [1, 2]),
                input(int_reg_32).eq(0),
                input(int_reg_32.or(m32).or(imm32)),
                G::Is0.output(flags.zf, [0]),
                G::Sign.output(flags.sf, [0]),
                // Unlike `Is0` and `Sign`, which are properties of the output, `overflow` and `carry`
                // are properties of the add operation itself, so they get names that reflect that
                // and take the add operands as parameters instead of the output (see the `[1, 2]`
                // instead of `[0]`).
                G::AddOverflow.output(flags.of, [1, 2]),
                G::AddCarry.output(flags.cf, [1, 2]),
                G::Parity.output(flags.pf, [0]),
            ]
            // TODO: Should this be defined per-output instead of per-instruction? Since an "output"
            //       should always act the same given the same inputs, no matter what instruction is
            //       used to implement it.
            .commutative(1, 2),
            // This is the encoding section. Unlike GCC which can just emit `add %0, %1, %2` and
            // let the assembler take care of checking that %0 and %1 match and handling the
            // difference between reg and mem operands, we have to do that work inline. We could
            // easily write helper functions so that this requires less code per-instruction.
            //
            // There are ways to use the type system to ensure that this closure receives arguments
            // of the right type, so this closure here would have arguments of type `Reg`, `Reg`,
            // `Mem32` (and receive no arguments for the various flags since they always go to the
            // same place) but the code to make this work is a little hairy so to keep things simple
            // I've written code that doesn't.
            |_must_be_same_as_b, b, c| match c {
                // Hand-waving away this "encode" module, this should just do
                // the task of actual assembly. This could even use `dynasm`
                // internally.
                Reg(c) => encode::add32_r_r(b.reg()?, c),
                // TODO: How to handle the fact that memory operands look different between
                //       backends? Maybe some output in `outputs::Generic` that converts
                //       some number of inputs into a memory operand? So you define a
                //       `G::ToMem` instr?
                Memory(/* TODO */) => encode::add32_r_m(b.reg()?, /* TODO */),
                Immediate(/* TODO */) => encode::add32_r_imm(b.reg()?, /* TODO */),
                _ => Err("Unexpected value"),
            },
        )
        .instr(
            [
                G::Mov32.output(m32, [1, 2]),
                int_reg_32.or(m32).or(imm32).input(),
            ],
            |a, b| match b {
                // Hand-waving away this "encode" module, this should just do
                // the task of actual assembly. This could even use `dynasm`
                // internally.
                Reg(c) => encode::mov32_r_r(b.reg()?, c),
                Memory(/* TODO */) => encode::mov32_r_m(b.reg()?, /* TODO */),
                Immediate(/* TODO */) => encode::mov32_r_imm(b.reg()?, /* TODO */),
                _ => Err("Unexpected value"),
            },
        )
}

/// A virtual register, used by Lightbeam's generic backend to track the flow of data
/// separate from the actual registers that are used on the machine.
struct VReg(usize);

/// A directive to the backend to generate an output from the given inputs
struct MachineIRInstr<S> {
    /// The name to give this specific output
    name: VReg,
    /// The output that we want
    output: Output<S>,
    /// The inputs that this output needs
    inputs: Vec<VReg>,
}

impl<S> Directive<S> {
    fn new(name: VReg, output: Output<S>, inputs: Vec<VReg>) -> Self {
        Directive {
            name,
            output,
            inputs,
        }
    }
}

/// A machine-specific backend for Lightbeam. This generates a series of `Directive`s,
/// basically a simple register machine that maps closely to how a real machine works,
/// but with register allocation and things like the `FLAGS` register abstracted away.
/// This allows us to write 90% of instruction implementations in a architecture-
/// agnostic way, with the remaining 10% either implemented with library calls or
/// using architecture-specific instructions.
trait LightbeamBackend {
    /// The machine specification type for this backend. This means that generic code
    /// can accept a `LightbeamBackend` and handle everything generically.
    type Machine: Machine;

    // `impl Trait` syntax is currently not implemented for traits but there are ways
    // around that limitation and it makes this code easier to read for now.
    //
    // We'd probably use `dyn Trait` to start with to prevent having many, many type
    // parameters, but there are definitely better ways. Probably the ideal solution
    // is to use generators, but those too are still unstable.
    //
    // The body of this function is defined in the trait itself - we can give default
    // implementations of any instruction that doesn't need anything machine-specific,
    // and theoretically for anything that _does_ need machine-specific instructions, we
    // can use a library of implementations in Rust that we call out to. This should
    // make it easy to add new backends, where replacing a slow implementation that
    // compiles to a function call with native code is as easy as overriding a function
    // in the trait impl.
    /// Add two numbers together
    fn add(
        regs: &mut impl Iterator<Item = VReg>,
        a: VReg,
        b: VReg,
    ) -> (VReg, Vec<Directive<Self::Machine::SpecificOutput>>) {
        let ret = regs.next();
        (
            ret,
            vec![Directive::new(ret, outputs::Generic::Add, vec![a, b])],
        )
    }
}
