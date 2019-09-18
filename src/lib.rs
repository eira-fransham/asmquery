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
//! -------------------------------------------------------------------------------------
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
//!       error
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
//! the carry bit. Obviously we should avoid generating patterns like this where possible, but it
//! means that if we have an `add` followed by an `add_carry` but DCE eliminates the `add`, we
//! still generate correct code.
//!
//! -------------------------------------------------------------------------------------
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
//! for example if the calculated address is needed later on. Here's an example of
//! what some Low IR with complex memory operations would look like. You'd write
//! Low IR that looks something like the code below and the algorithm above would
//! collapse all of these into a single instruction on x86 but multiple
//! instructions on ARM64 (assume `%base`, `%index` and `%lhs` were defined
//! previously):
//!
//! ```
//! %mem0 = add %base, %disp
//! %shifted_index = shl %index, 2imm3
//! %mem1 = add %mem0, %shifted_index
//! %loaded = load %mem1
//! %added = add %lhs, %loaded
//! ```
//!
//! We'd probably write some helper method for x86 that abstracted this away.
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
//! A limitation of this is that if you need to use the same address twice, as far
//! as I can tell it isn't possible to write a single stream of Low IR instructions
//! that would do the address calculation once and save it to an intermediate
//! result on ARM but do the calculation twice on x86, since doing the calculation
//! twice would be slower on ARM but storing to an intermediate register would be
//! slower on x86. Probably we can just delegate that responsibility to the Low IR
//! generator.
//!
//! -------------------------------------------------------------------------------------
//!
//! It might be useful to maintain a bitmask for some subset of outputs that represents
//! the set of instructions that can actually produce that value into a specific
//! register or memory location. For example, `add reg, reg` produces an `Add32` output
//! into location that can be later accessed, whereas after emitting
//! `mov r2, [r0 + r1]` you cannot access `r0 + r1`. We still want to keep the `mov` as
//! a possible candidate in case the next Low IR operation is a `load` so we can't just
//! avoid producing it as a match in the first place. If we have some kind of cache
//! with (probably precalculated) bitmasks based on target locations, we can
//! pretty easily mask out any instructions that we know for sure are going to be
//! invalid and only iterate over the remaining ones. This is especially the case for
//! x86 memory addressing - many, many, many x86 instructions have variants that take a
//! memory operand, so if we query for an "add" or "shift" we'll get a huge number of
//! false positives. We could precache "instructions that can put an add result in any
//! GPR", which will create a bitmask for every instruction is at least one `Add32`
//! output whose set of possible destinations has any overlap at all with the set of
//! GPRs.
//!
//! -------------------------------------------------------------------------------------
//!
//! Clobbers can just be represented as other outputs. A clobber that zeroes some part of
//! a register becomes a `Zero` operation parameterised with the correct size, likewise a
//! clobber that leaves some register in an undefined state could be represented as an
//! `Undefine` operation. The same code that prevents a register from being overwritten by
//! an intended output can also be used to prevent a register from being overwritten by an
//! unintended output. This is one of the biggest places that we have bugs right now so
//! factoring all of the clobber avoidance code into a single place will make codegen far
//! more robust.
//!
//! -------------------------------------------------------------------------------------
//!
//! One thing that is useful to note is that virtual registers should map one-to-one with
//! something that actually exists on the machine. We should never reallocate a register,
//! if it needs to be moved a new register should be allocated. However, we _should_ be
//! able to reallocate anything that was pushed onto the Wasm stack. The internal
//! representation of the stack should be split into a vector of virtual registers and a
//! mapping from virtual register to real location. We may also want a mapping the other
//! way around - from real locaton to virtual register and then from virtual register to
//! positions on the Wasm stack. A mapping that goes the opposite direction means that
//! we can efficiently free up registers without having to iterate over the entire stack.
//!
//! -------------------------------------------------------------------------------------
//!
//! How to model control flow is the biggest question-mark here, as it often is. Although
//! this model works great for straight-line code there are complexities when it comes to
//! modelling any control flow. It might be useful for the methods on the Low IR
//! generator that implement control flow instructions to recieve information like the
//! target calling convention so it can emit `mov`s etc that directly implement this.
//! Having Low IR implement control flow is desirable - I would ideally want to prevent
//! LIRC from generating any Low IR itself whatsoever, even delegating the implementation
//! of `mov`s to the machine-specific generator, but this might not be useful.
//!
//! Something that is cross-purposes with control flow: how do we handle conditional
//! instructions? ARM64 has conditional increment, conditional not, conditional negate,
//! and conditional move, whereas x86 has conditional move and conditional set. We
//! definitely want to at least support conditional move, since there is a Wasm
//! instruction that maps directly to it (`select`).
//!
//! The simplest solution is to just have `CMov` be a separate output, one that takes 3
//! inputs - along with the src and dst we can additionally provide a condition. Ideally,
//! though, we would somehow combine control flow and conditional instructions, since
//! that would mean that we could compile code in Wasm that uses control flow to skips
//! over instructions to use conditional instructions on the target architecture.
//! Perhaps, when hitting control flow where one branch is directly following the current
//! one, we can delay generating the actual branch, only doing so if we hit an instruction
//! that cannot be made conditional. This would end up being pretty hairy though, of
//! course, since we'd have to avoid clobbering any flags etc that the jump would need.
//!
//! -------------------------------------------------------------------------------------
//!
//! An idea for how to handle calling conventions and control flow is as follows: we have
//! a concept of calling conventions in the IR that are defined in terms of virtual
//! registers. Since virtual registers must be globally unique (i.e. you can't redefine
//! them even in distinct codepaths) we can have each block simply define the virtual
//! registers that it needs to be live when you enter it, plus a list of arguments for
//! locations that can be different every time the block is entered. The Low IR would
//! then define a calling convention and apply it to some number of blocks:
//!
//! ```
//! .newcc sharedcc (%bar) [%something]
//!
//!   %something = const i32 1
//!   %condition = is_zero %somereg
//!   %foo = const i32 1
//! .applycc sharedcc (%foo)
//!   jmpif %condition, true_branch
//!   jmp false_branch
//! label true_branch sharedcc:
//!   ;; ...
//! label false_branch sharedcc:
//!   ;; ...
//! ```
//!
//! The registers in the `[]` are dependencies - registers that must be live when the
//! block is entered. There are no restrictions on these registers, for example, they can
//! be constants. Registers in the `()` are arguments - these are passed to the block
//! every time it is called. Since arguments can be different, when the block is first
//! called a mutable location is allocated for it - normally a register. This means that
//! if we pass a constant as an argument we have to spill that constant to a register.
//!
//! This maintains the property that every virtual register must correspond to precisely
//! one location on the machine. For a block that has only one caller, the Low IR
//! generator can create a calling convention that has no arguments.
//!
//! We can also use this system to implement calls, so long as we allow the Low IR to
//! specify arguments as either physical locations or as virtual registers. You could
//! imagine that the Low IR might look something like so:
//!
//! ```
//! .newcc systemvi32_i32 (%rsi, %rdi) []
//! .newcc systemvreti32 (%rax) []
//!
//!   %foo = const i32 0
//!   %bar = const i32 1
//!   %funcpointer = get_function_pointer_somehow
//! .applycc systemvi32_i32 (%foo, %bar)
//!   ;; TODO: Exactly how a call looks isn't clear right now
//!   jmp some_function
//! label return_from_call systemvreti32:
//!   ;; ...
//! ```
//!
//! You can see here that we actually define a new block that would be executed after
//! `some_function` returns. You can see that the fact that `some_function` returns by
//! calling `return_from_call` is implicit, based on the fact that `return_from_call` is
//! directly after the function call. If we wanted to have dead code elimination of any
//! kind we'd have to model this better. The cleanest way to solve this issue would be by
//! splitting the `call` instruction into its components, so a call would be calculating
//! the offset between the `return_from_call` label and the current instruction pointer,
//! push that to stack, and then branch. However, because we only have one-instruction
//! lookahead we can't do this. So probably if we ever want to implement DCE at the
//! level of Low IR we could just have an assembler directive that explicitly marks a
//! label as used.
//!
//! Something we would probably want to do is have a spill instruction that specifies
//! what is off-limits, and then emit a spill instruction for each variable that we want
//! to be maintained across the boundary of the function. For example:
//!
//! ```
//! ;; .. snip..
//! .newcc systemvreti32 (%rax) [%keep_me, %keep_me_too]
//!
//!   ;; ..snip..
//!   %something = const i32 1
//!   %something_else = ;; some calculation that produces its value in `%rsi`
//!   %keep_me = spill %something, [%rsi, %rdi, ..]
//!   %keep_me_too = spill %something_else, [%rsi, %rdi, ..]
//!   ;; ..snip..
//! .applycc systemvi32_i32 (%foo, %bar)
//!   ;; ..snip..
//! label return_from_call systemvreti32:
//!   %new_variable = add %keep_me, %keep_me
//!   ;; ...
//! ```
//!
//! In this case we can see that `%keep_me` would be exactly the same as `%something`
//! because it doesn't overlap with any of the locations in the square brackets, whereas
//! %keep_me_too` would be different to `%something_else`. This same `spill` system can
//! be used when we need a specific register for e.g. `div`, simply emitting a `spill`
//! before we emit the `div`. Although I've written the list of banned locations inline
//! here, this will probably be implemented by having a single register class for each
//! of the kinds of spilling we want to do (systemv calls, `div` instructions, etc) and
//! just referencing them.
//!
//! -------------------------------------------------------------------------------------
//!
//! When we branch or do anything that uses a label, we want to be able to have a
//! location that we can write to with the actual value of that label. Since in the
//! current design, every instruction definition represents precisely one encoding of the
//! instruction, we could have a system where we simply find out how much space we need
//! for the instruction, then we take a note of which instruction definition we need to
//! encode, what the arguments are (not including the ones we'll fill in later) and what
//! our current encoding position is. When that label gets defined, we simply call back
//! into the encoding function of that specific instruction definition and overwrite the
//! whole instruction.
//!
//! To ensure that we don't accidentally write an instruction definition that can return
//! encodings of different sizes depending on the arguments, we could have it so that the
//! method to define a new instruction enforces that the supplied encoding function
//! returns a fixed-size array (probably using a trait). We then don't even need to even
//! call the function when we have relocations, we just get the size and fill it in with
//! zeroes. That way, the assembler doesn't even need to know about the concept of
//! relocations whatsoever.
//!
//! -------------------------------------------------------------------------------------
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
///
/// TODO: We should be able to specify constant folding rules for `Output`s, and just add a
///       step to the algorithm that greedily folds constants together if both arguments
///       are constants. This should be far, far less error-prone than the current constant
///       folding system.
enum Output<S> {
    Generic(outputs::Generic),
    Specific(S),
}

// TODO: Handle instructions that don't return a value, like `jmp`, `return`, `ud2` etc.
//       Probably this can be handled like other outputs, but where the output is of
//       the bottom type (i.e. a dummy invalid type) and so matches all constraints.
struct Query<S>(Output<S>, Constraint, Vec<Constraint>);

/// A structure that represents a list of candidate instructions that can be refined by
/// applying further queries. This should be a bitfield of some kind.
struct Matches<S> {/* ..TODO.. */}

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

    // When we define `R0` etc, we should specify its size in bits
    let int_reg_64 = RegClass::new([R0, R1, R2 /* ..snip.. */]);
    // `r32` should be precisely equal to `.mask(0xFFFF)`
    let int_reg_32 = RegClass::new([r32(R0), r32(R1), r32(R2) /* ..snip.. */]);
    let float_reg_64 = RegClass::new([/* ..snip.. */]);
    let imm32 = Immediate(32);

    // TODO: How to handle some registers overlapping with others? For example, some 32-bit
    //       instrs clobber the whole 64-bit reg, others leave the upper bits untouched.
    //       Maybe we just don't handle it at all for now, since although it could lead to
    //       better codegen it's not necessary for correctness AFAIK.
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

    // Since we want the set of outputs to be as minimal as possible, we'd ideally have equality defined
    // as `is_zero` on the output of a `sub`. However, having to rewrite that and remember that you have
    // to implement equality that way in every new machine spec is tedious. What we really want is to
    // have a way to abstract this way. My idea is to have each element of the output array be an
    // iterator. As far as I know, there's no harm in specifying _both_ an `eq` output and a `sub`-with-
    // `is_zero` output that just point to the same destination - LIRC should still be able to easily
    // handle that correctly - but it's error-prone if we have to specify that every single time, when
    // we know that it will always be true that `a - b == 0` is the same as `a == b`.
    MachineSpec::new()
        .instr(
            // This is the metadata section. This specifies everything that the query engine needs,
            // so inputs, outputs, clobbers, commutativity, maybe even identity.
            [
                // These are the parameters that need to be reused in several outputs. To ensure that
                // we only generate a single register for them we specify them literally here. You can
                // only define outputs whose destination and inputs are references to inputs,
                // references to other outputs or individual registers.
                input(int_reg_32),
                input(int_reg_32),
                G::Add(32).output(Ref(0).mask(0x0000FFFF), [Ref(0), Ref(1)]),
                // TODO: Is this really necessary? You could imagine it being used to implement constant-
                //       folding when we do some kind of work with the upper bits of a zeroed register.
                //
                //       For now I think we should ignore the mask when trying to work out whether a
                //       register is clobbered, and only care about it in simple cases like `FLAGS`.
                G::Zero.output(Ref(0).mask(0xFFFF0000)),
                G::Is0.output(flags.zf, [Ref(2)]),
                G::Sign.output(flags.sf, [Ref(2)]),
                G::Parity.output(flags.pf, [Ref(2)]),
                // Unlike `Is0`/`Sign`/`Parity`, which are properties of the output, `overflow` and
                // `carry` are properties of the add operation itself, so they get names that reflect
                // that and take the add operands as parameters instead of the output (see the `[Ref(0),
                // Ref(1)]` instead of `[Ref(2)]`). These are outputs that essentially represent whether
                // this operation _would_ overflow were you to perform it, which is useful because
                // something like `cmp` can tell you if subtraction will underflow without actually
                // performing a subtraction. If you've requested a `SubUnderflow32` but not used the
                // actual result of the `sub`, we can emit a `cmp` instead.
                G::AddOverflow(32).output(flags.of, [Ref(0), Ref(1)]),
                G::AddCarry(32).output(flags.cf, [Ref(0), Ref(1)]),
            ],
            // This is the encoding section. Although we need to write a different instruction definition
            // for every form of the instruction (reg-reg, reg-mem, reg-imm, etc) we can trivially write
            // a helper that has some higher-level form and converts it to multiple instruction
            // definitions.
            |args| encode::add32_r_r(args.at(0).reg()?, args.at(1).reg()?),
        )
        .instr(
            // Since both adds are commutative in this instruction we'd need to generate several versions
            // of it with the parameters in different orders.
            [
                input(int_reg_32), // DEST
                input(imm32),      // DISP
                input(int_reg_32), // BASE
                input(int_reg_32), // INDEX
                input(imm3),       // SCALE
                // load(BASE + (INDEX << SCALE) + DISP)
                G::Load32.output(Ref(0), [Ref(6)]),
                // BASE + (INDEX << SCALE) + DISP
                G::Add32.output(INTERNAL, [Ref(2), Ref(7)]),
                // (INDEX << SCALE) + DISP
                G::Add32.output(INTERNAL, [Ref(1), Ref(8)]),
                // INDEX << SCALE
                G::ShiftL32.output(INTERNAL, [Ref(3), Ref(4)]),
            ],
            |args| {
                let dest = args.at(0);
                let disp = args.at(1);
                let base = args.at(2);
                let index = args.at(3);
                let scale = args.at(4);

                encode::mov32_r_m(dest.reg()?, Mem32(base, index, scale, disp))
            },
        )
        // With x86 jumps, there are instructions that do both a logical op and the jump itself.
        // `ja` is defined to jump if CF = 0 _and_ ZF = 0. It also has an instruction to jump
        // if `ECX` is 0.
        .instr(
            [
                input(int_reg_32),
                // TODO: Should this be an actual output or should we have a separate category for
                //       operations that don't produce anything? Or, should we represent this as a
                //       conditional move into `rip`?
                G::JumpIf.output(.., [Ref(2), Ref(0)]),
                // Even though the internal `And` is commutative, we don't need to generate two versions
                // of this instruction because both these inputs are fixed.
                G::And(1).output(INTERNAL, [flags.cf, flags.zf]),
            ],
            |a, b| { /* TODO */ },
        )
}

/// A virtual register, used by Lightbeam's generic backend to track the flow of data
/// separate from the actual registers that are used on the machine.
struct VReg(usize);

/// A single instruction in Low IR. This is a RISC instruction set where each `Output`
/// specification corresponds to multiple instructions that could theoretically
/// produce that output. See module documentation for a better description.
struct LowIRInstr<S> {
    /// The name to give this specific output
    name: VReg,
    /// The output that we want
    output: Output<S>,
    /// The inputs that this output needs
    // TODO: This should include a "dead value" flag to explicitly mark the final use
    //       of a value. This makes location reuse explicit (replacing the ad-hoc
    //       `take`/`release` system in Lightbeam's current backend) and since every
    //       value has a unique ID we can have debug-mode assertions that ensure that
    //       we have no double-frees or dangling values.
    inputs: Vec<VReg>,
}

impl<S> LowIRInstr<S> {
    fn new(name: VReg, output: Output<S>, inputs: Vec<VReg>) -> Self {
        LowIRInstr {
            name,
            output,
            inputs,
        }
    }
}

/// A machine-specific backend for Lightbeam. This generates a series of `LowIRInstr`s,
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
    ) -> (VReg, Vec<LowIRInstr<Self::Machine::SpecificOutput>>) {
        let ret = regs.next();
        (
            ret,
            vec![LowIRInstr::new(ret, outputs::Generic::Add, vec![a, b])],
        )
    }
}
