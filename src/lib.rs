//! This is a sketch of how a vertical slice through the x86 assembly pipeline might look
//! like.
//!
//! Here's a quick overview/glossary:
//! - I'm going to mostly use "instruction" to refer to the concept of instruction
//!   _definitions_ in the machine spec. This will mostly line up one-to-one with
//!   opcodes in the ISA but there will probably be some architectures for which
//!   multiple opcodes are needed to generate a single output, in which case all
//!   those opcodes together will be one instruction definition. For simplicity I'm
//!   just going to use the terms "instruction definition" and "instruction" interchangably,
//!   but when I refer to a query returning a list of "instructions" each "instruction" may
//!   not technically encode to a single opcode on the machine.
//!
//! - The biggest difference between this and other similar libraries is that this doesn't
//!   have any outwards-facing concept of "instructions". Instead, Lightbeam requests an
//!   "output" and the assembler produces a (TODO: lazy?) list of possible instructions,
//!   allowing Lightbeam to request _multiple_ outputs and the assembler will try to
//!   return one instruction that produces all of them. For example, the machine-specific
//!   backend can request an add-with-carry, which might look like this (strawman syntax):
//!   ```
//!   %2 = add %0, %1
//!   %3 = carry %2
//!   ```
//!   This means that results that we get as a side-effect and results we get because we
//!   specifically asked for them are not distinguished. I'm pretty sure this will make code
//!   simpler on the compiler-side, since we can have a single codepath that handles both.
//!
//!   For an example of how this may simplify things, it means that the compiler can
//!   write code generically handling the case that we already have an is-zero
//!   result from doing an add and the case that we need to emit a `test`. We can even
//!   allow the compiler to request "32-bit add with carry output", "32-bit add and
//!   I don't need the carry output" and "32-bit add that cannot clobber the carry bit"
//!   without special-casing each as different instructions.
//!
//! - This is diverging from GCC, but I use "constraint" to refer to a component of an
//!   instruction query, where it refers to the query specifying that a specific argument
//!   should be in a specific place. For the part of the machine specification that
//!   specifies when a constraint should match I will use "predicate". Instructions _only_
//!   have predicates, which represent both the kind of location (register, memory, etc)
//!   and also anything that must further be satisfied, such as an instruction needing
//!   its input in specific registers.
//!
//! - When I say a "directive" I mean an instruction such as the above - a single output
//!   assigned to a virtual register, with a list of virtual registers as input. Directives
//!   are a concept that exists only in Lightbeam, as they are tied to the register allocator,
//!   value stack, etc. The assembler only works in outputs, which may have inputs with
//!   constraints. Directives are generated by the machine-specific backend and converted
//!   by the machine-generic backend into queries.
//!
//! - When I refer to "the assembler" or "this library" I'm talking about the code that
//!   converts possibly-machine-specific outputs into definitely-machine-specific bytes
//!   of machine code. I'll try to consistently use "the assembler" since this page
//!   contains some code as an example of how Lightbeam might use the assembler and
//!   saying "this library" might be confusing
//!
//! - When I refer to  the "machine-generic backend" I'm referring to the code that calls
//!   into the assembler. It's the machine-generic backend's responsibility to allocate
//!   specific registers and track the fact that generating some other outputs (for example,
//!   another `add`) between that `add` and `carry` may make the information that `carry`
//!   needs inaccessible, and the machine-specific backend's responsibility to not generate
//!   a stream of directives that cannot be fulfilled by the assembler.
//!
//! - When I refer to the "machine-specific backend", I'll be referring to the code written
//!   for each target architecture that maps the Microwasm instructions to a stream of
//!   directives, which will then be used by Lightbeam's machine-generic code to query for
//!   specific instructions.
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
//!   with a stateless machine-specific directive generator `G`
//! * Generate a stateless machine-specific assembler `A`
//! * For each WebAssembly instruction (call each instruction `WI`)
//!   * Convert `WI` to a stream of Microwasm instructions `MI`
//!   * Convert `MI` into a stream of directives `DS`
//!     * Each individual Microwasm instruction in `MI` is converted into a stream of
//!       directives by `G` and these streams are concatenated together
//!   * Create a stateful cursor `CD` into `DS`
//!   * While `CD` has not reached the end of `DS`
//!     * Get the current directive `D` pointed to by `CD` and advance the cursor
//!     * `B` fills in the virtual registers in `D` with constraints based on its
//!       internal state to make a query `Q`
//!     * `B` passes `Q` into the assembler to get a list of matches `M`
//!     * While `M` is not empty
//!       * Get the current directive `D'` pointed to by `CD` and advance the cursor
//!       * `B` fills in the virtual registers in `D'` with constraints based on its
//!         internal state to make a query `Q'`
//!       * Refine `M` to only contain the matches that _also_ match `Q'` (NOTE: see below)
//!     * Get the best match in `M` (by some definition of "best", perhaps by which
//!       match requires the least spilling or even by cycle count). Note that this match
//!       is not necessarily a single instruction
//!     * Pass this best match to the assembler to encode the output into a byte buffer
//!
//! > NOTE: Constantly iterating over `M` to refine it will probably kill our linear-time goal -
//!         I believe that technically it's still O(n) since the number of instructions per
//!         backend is constant but the number of instructions could be quite high. What we
//!         really want is to have as much refinement as possible done at compile-time, and
//!         since some of the constraints (specifically, the outputs and the flow of data) are
//!         known at compile-time, we could concievably have some of this work done in advance
//!         so that the refinement has a far smaller set to handle. Quite how this will work
//!         isn't clear, though. We might have to have some abstraction in the assembler query
//!         engine where it can handle data flow so long as no supplementary registers need to
//!         be allocated, or even that it can handle allocating some registers with those
//!         registers being classed as "clobbers" as far as Lightbeam is concerned. This could
//!         allow us to move far more work into compile-time, at the expense of complexity in
//!         the assembler and a blurring of the lines between backend and assembler.
//!
//! A cool thing about this algorithm: assuming that it can be efficiently implemented this
//! gives us optimisations like converting add-and-test into add-and-use-side-effect-flags
//! while _also_ converting add-without-test into `lea` where appropriate, without explicitly
//! implementing it or even needing to track what side-effects have been produced but not
//! clobbered. More optimisations could be implemented with better tracking of flags in the
//! generic backend - such as `add, mov, test` being converted into `add, mov` with the test
//! implicit in the `add` and not clobbered by the `mov` - but it's great that we get some
//! optimisations implemented for free. This algorithm also implicitly prevents patterns like
//! the following:
//!
//! ```
//! %2 = add %0, %1
//! %3 = sub %1, %2
//! %4 = carry %2
//! ```
//!
//! Since the `carry` output can't be combined with the `sub` output, a query will be performed
//! for `carry` alone, but since there is no instruction that has a `carry` output for a
//! register provided as input (the only instructions with a `carry` output are outputting the
//! `carry` for _other_ outputs) it will cause no instructions to be found and the compiler will
//! error out. However, sequences like the following _will_ succeed, since although we overwrite
//! `zf` with` sub` when we query the assembler for `is_zero` it can still return `test`:
//!
//! ```
//! %2 = add %0, %1
//! %3 = sub %1, %2
//! %4 = is_zero %2
//! ```
//!
//! > TODO: What does the querying API look like? Do we need constraints at all? We definitely
//!         need some way to specify the flow of data so that patterns like the above work,
//!         but actually constraining them to actual locations seems like it should be
//!         unnecessary. Should we have `mov` (such as register-register, or register-immediate)
//!         be a separate instruction to `store`/`load` (memory-register/register-memory) or is
//!         it better to have the latter expressed as `mov` with a constraint that one operand
//!         must be a memory location? For now I've chosen the latter.
//!
//! The lowest level is the actual machine specification, which allows for both the
//! generic outputs that we should expect every machine to have and the machine-specific
//! outputs that implement behaviour that is unique to each machine. This allows
//! Lightbeam to provide default implementations of many Wasm instructions to most
//! backends using the generic outputs, while still making the Rust compiler complain
//! for any Wasm instructions that can't be implemented using outputs that are in
//! every machine. At some point we might find that the concept of a generic instruction
//! for every machine in existence is unhelpful and that we want to drill down and have
//! some concept of outputs that are generic across some subset of machines, but
//! for now it's far simpler to have most outputs exist for all machines.
//!
//! Everywhere where we use `Vec` we'd ideally use some trickery to do everything on the
//! stack. I have ideas of precisely how to constrain ourselves to the stack but to keep
//! this sample code simple I've used `Vec`.

/// An argument constraint, this means that this input or output must be in a specific place.
struct Constraint {
    reg: bool,
    mem: bool,
    imm: bool,
}

/// Any value that can be obtained from another value. This is an "output" and not an
/// "instruction", as there is a many-to-many relationship between outputs and instructions -
/// instructions can have many outputs, and a single output could be generated from any
/// number of different instructions.
enum Output<S> {
    Generic(outputs::Generic),
    Specific(S),
}

/// A virtual register, used by Lightbeam's generic backend to track the flow of data
/// separate from the actual registers that are used on the machine.
struct VReg(usize);

/// A virtual value, which has a unique identifier so that the query engine can correctly
/// refine a series of queries into a single instruction that correctly preserves data
/// flow, and a constraint on its location. For most instructions it's not important where
/// one argument should be, so long as it can be moved there. For others one operand _must_
/// be in a specific place. As far as I know, the only instructions for which this is
/// absolutely true are `mov`-family instructions, since it doesn't matter if we only have
/// register-register add and we need it in a memory location so long as we can move it to
/// that memory location afterwards.
struct Value {
    name: VReg,
    constraint: Constraint,
}

// TODO: Handle instructions that don't return a value, like `return` or `ud2`.
//       Probably this can be handled like other outputs, but where the output is of
//       the bottom type (i.e. a dummy invalid type).
// TODO: This doesn't handle clobbers, which
struct Query<S>(Output<S>, Value, Vec<Value>);

/// Any structure that represents a list of candidate instructions that can be refined by
/// applying further queries
trait Matches<S> {
    fn refine(&mut self, query: Query<S>);
}

trait Machine {
    /// Type of machine-specific outputs
    type SpecificOutput;
    /// The specific structure implementing the query engine for this machine
    ///
    /// > TODO: Do we need this to be a type parameter or can we have this be one type
    ///         that works on all machines? Almost certainly the latter.
    type Matches: Matches<Self::SpecificOutput>;

    /// Machines should be stateless, so we use `&` here, but it's possible that
    /// we might want to use `&mut` in the future.
    fn query(&self) -> Self::Matches;
}

// I've chosen to write an embedded DSL here, so that we don't have to write
// blit-a-string-to-a-file-style codegen but we could totally have this be an
// external DSL like in GCC instead. Writing the codegen would be more of a
// hassle and make compile-times worse than if we use an eDSL though.
//
// This will (or rather, should) compile to code as good as hand-rolled code.
// We currently use values for most things instead of encoding things into
// the type system, but using `const fn` should make it so that we can have
// most of our code compile into the best possible code.
//
// TODO: How do we handle control flow? The backend should only expose very
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
//       a memory location, where the input is moved, and `rsp`, which is
//       incremented by 8 respectively. Lightbeam just needs to request the
//       value to be moved and the stack pointer to be incremented by 8 and
//       the query refinement algorithm will take care of the rest.
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
                // Even though this is a 32-bit add I've used a 64-bit register, since a 32-bit add
                // will zero the upper bits of the register. Not sure if we could have a better way
                // of specifying this.
                G::Add32.output(int_reg_64, [1, 2]),
                input(int_reg_32).eq(0),
                input(int_reg_32.or(m32).or(imm32)),
                G::Is0.output(flags.zf, [0]),
                G::Sign.output(flags.sf, [0]),
                G::Overflow.output(flags.of, []),
                G::Carry.output(flags.cf, []),
                G::Parity.output(flags.pf, [0]),
            ]
            .commutative(1, 2),
            // This is the encoding section. Unlike GCC which can just emit `add %0, %1, %2` and
            // let the assembler take care of checking that %0 and %1 match and handling the
            // difference between reg and mem operands, we have to do that work inline. We could
            // easily write helper functions so that this requires less code per-instruction.
            |_must_be_same_as_b, _must_be_flags, b, c| match c {
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
                // TODO: How to handle the fact that memory operands look different between
                //       backends? Maybe some output in `outputs::Generic` that converts
                //       some number of inputs into a memory operand? So you define a
                //       `G::ToMem` instr?
                Memory(/* TODO */) => encode::mov32_r_m(b.reg()?, /* TODO */),
                Immediate(/* TODO */) => encode::mov32_r_imm(b.reg()?, /* TODO */),
                _ => Err("Unexpected value"),
            },
        )
}

/// A directive to the backend to generate an output from the given inputs
struct Directive<S> {
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
