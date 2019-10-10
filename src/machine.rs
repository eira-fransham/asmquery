use generic_array::{sequence::GenericSequence, ArrayLength, GenericArray, IntoArray};
use smallbitvec::SmallBitVec;
use std::{fmt, io};

/// The only important thing about registers is that they don't overlap, so we can just use an
/// opaque ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reg {
    /// The opaque ID of this register
    id: u64,
}

impl Reg {
    pub const fn from_id(id: u64) -> Self {
        Reg { id }
    }

    pub const fn id(&self) -> u64 {
        self.id
    }
}

#[macro_export]
macro_rules! regs {
    ($name:ident) => {
        const $name: $crate::machine::Reg = $crate::machine::Reg::from_id(0);
    };
    ($first:ident, $second:ident $(, $rest:ident)*) => {
        regs!($second $(, $rest)*);
        const $first: $crate::machine::Reg = $crate::machine::Reg::from_id($second.id() + 1);
    };
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RegClass<'a>(pub &'a [Reg]);
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Immediate {
    pub bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Bound<'a> {
    Reg(RegClass<'a>),
    Imm(Immediate),
}

impl<'a> From<RegClass<'a>> for Bound<'a> {
    fn from(other: RegClass<'a>) -> Self {
        Bound::Reg(other)
    }
}

impl From<Immediate> for Bound<'_> {
    fn from(other: Immediate) -> Self {
        Bound::Imm(other)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Var {
    id: usize,
}

impl Var {
    pub fn id(&self) -> usize {
        self.id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param<'a> {
    pub var: Var,
    pub bound: Bound<'a>,
}

#[derive(Debug)]
pub enum EncodeArg {
    Reg(Reg),
    // TODO: Should we increase the size of this and/or make it generic?
    Imm(u128),
}

#[derive(Debug)]
pub struct EncodeError;

pub type EncodeResult = Result<(), EncodeError>;

type ParamIter<'borrow, 'a> = impl Clone + Iterator<Item = Param<'a>> + 'borrow;
type ActionIter<'borrow, T> = impl Clone + Iterator<Item = &'borrow Action<T>> + 'borrow;
type EqualityIter<'borrow> = impl Clone + Iterator<Item = (Var, Var)> + 'borrow;

pub struct InstrDef<'borrow, 'a: 'borrow, T: 'borrow> {
    name: &'a str,
    params: ParamIter<'borrow, 'a>,
    actions: ActionIter<'borrow, T>,
    equality: EqualityIter<'borrow>,
}

impl<'borrow, 'a, T> InstrDef<'borrow, 'a, T> {
    pub fn name(&self) -> &'a str {
        self.name
    }

    pub fn params(&self) -> impl Iterator<Item = Param<'a>> + 'borrow {
        self.params.clone()
    }

    pub fn actions(&self) -> impl Iterator<Item = &'borrow Action<T>> {
        self.actions.clone()
    }

    pub fn equality(&self) -> impl Iterator<Item = (Var, Var)> + 'borrow {
        self.equality.clone()
    }

    pub fn map_encode(&self, _args: &[EncodeArg], _buf: &mut dyn io::Write) -> EncodeResult {
        unimplemented!()
    }
}

#[derive(Debug, Clone)]
struct InstrDefInternal<'a> {
    name: &'a str,
    params: (usize, SmallBitVec),
    actions: (usize, SmallBitVec),
    equality: (usize, SmallBitVec),
}

#[derive(Debug)]
pub struct MachineSpec<'a, T> {
    params: Vec<Param<'a>>,
    actions: Vec<Action<T>>,
    equality: Vec<(Var, Var)>,
    instrs: Vec<InstrDefInternal<'a>>,
}

impl<T> Default for MachineSpec<'_, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Display for MachineSpec<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for instr in self.instrs_iter() {
            writeln!(f, "{}:", instr.name())?;
            writeln!(
                f,
                "  PARAMS: {:?}",
                instr.params().map(|p| p.var.id()).collect::<Vec<_>>()
            )?;
            writeln!(f, "  ACTIONS:")?;
            for (l, r) in instr.equality() {
                writeln!(f, "    {} = {} ", l.id(), r.id())?;
            }
            for Action {
                dest,
                action,
                inputs,
            } in instr.actions()
            {
                writeln!(
                    f,
                    "    {} = {:?} {:?}",
                    dest.id(),
                    action,
                    inputs.iter().map(|v| v.id()).collect::<Vec<_>>()
                )?;
            }
        }

        Ok(())
    }
}

impl<'a, T> MachineSpec<'a, T> {
    pub fn new() -> Self {
        MachineSpec {
            equality: vec![],
            params: vec![],
            actions: vec![],
            instrs: vec![],
        }
    }

    pub fn instrs_iter(&self) -> impl Iterator<Item = InstrDef<'_, 'a, T>> + '_ {
        fn param_type_alias_hack<'borrow, 'a>(
            params: &'borrow [Param<'a>],
            bitvec: &'borrow SmallBitVec,
        ) -> ParamIter<'borrow, 'a> {
            params[..bitvec.len()]
                .iter()
                .enumerate()
                .filter(move |(i, _)| bitvec[*i])
                .map(|(_, v)| v.clone())
        }

        fn actions_type_alias_hack<'borrow, T>(
            actions: &'borrow [Action<T>],
            bitvec: &'borrow SmallBitVec,
        ) -> ActionIter<'borrow, T> {
            actions[..bitvec.len()]
                .iter()
                .enumerate()
                .filter(move |(i, _)| bitvec[*i])
                .map(|(_, v)| v)
        }

        fn equality_type_alias_hack<'borrow>(
            equality: &'borrow [(Var, Var)],
            bitvec: &'borrow SmallBitVec,
        ) -> EqualityIter<'borrow> {
            equality[..bitvec.len()]
                .iter()
                .enumerate()
                .filter(move |(i, _)| bitvec[*i])
                .map(|(_, v)| v.clone())
        }

        self.instrs.iter().map(move |instr| InstrDef {
            name: instr.name,
            params: param_type_alias_hack(
                if instr.params.1.is_empty() {
                    &[]
                } else {
                    &self.params[instr.params.0..]
                },
                &instr.params.1,
            ),
            actions: actions_type_alias_hack(
                if instr.actions.1.is_empty() {
                    &[]
                } else {
                    &self.actions[instr.actions.0..]
                },
                &instr.actions.1,
            ),
            equality: equality_type_alias_hack(
                if instr.equality.1.is_empty() {
                    &[]
                } else {
                    &self.equality[instr.equality.0..]
                },
                &instr.equality.1,
            ),
        })
    }

    pub fn instr<F>(self, name: &'a str, func: F) -> Self
    where
        F: FnOnce(&mut InstrBuilder<'a, T>),
    {
        let mut builder = InstrBuilder::new(name);

        func(&mut builder);

        self.merge(builder.inner)
    }

    fn merge(mut self, other: MachineSpec<'a, T>) -> Self {
        let params_offset = self.params.len();
        let actions_offset = self.actions.len();
        let equality_offset = self.equality.len();

        self.params.extend(other.params);
        self.actions.extend(other.actions);
        self.equality.extend(other.equality);
        self.instrs
            .extend(other.instrs.into_iter().map(|instr| InstrDefInternal {
                name: instr.name,
                params: (params_offset + instr.params.0, instr.params.1),
                actions: (actions_offset + instr.actions.0, instr.actions.1),
                equality: (equality_offset + instr.equality.0, instr.equality.1),
            }));

        self
    }
}

#[derive(Default, Debug, Clone)]
struct VariableBuilder {
    cur: usize,
}

impl VariableBuilder {
    fn next(&mut self) -> Var {
        let out = self.cur;
        self.cur += 1;
        Var { id: out }
    }
}

pub struct InstrBuilder<'a, T> {
    name: &'a str,
    variable_builder: VariableBuilder,
    inner: MachineSpec<'a, T>,
}

impl<'a, T> InstrBuilder<'a, T> {
    fn new(name: &'a str) -> Self {
        let mut inner = MachineSpec::new();
        inner.instrs.push(InstrDefInternal {
            name,
            params: (0, Default::default()),
            actions: (0, Default::default()),
            equality: (0, Default::default()),
        });
        InstrBuilder {
            name,
            variable_builder: Default::default(),
            inner,
        }
    }
}

impl<'a, T> InstrBuilder<'a, T>
where
    T: Clone,
{
    pub fn var(&mut self) -> Var {
        self.variable_builder.next()
    }

    pub fn param(&mut self, bound: impl Into<Bound<'a>>) -> Var {
        let bound = bound.into();
        let var = self.variable_builder.next();

        for InstrDefInternal {
            params: (offset, mask),
            ..
        } in &mut self.inner.instrs
        {
            if mask.is_empty() {
                *offset = self.inner.params.len();
            }
            while *offset + mask.len() < self.inner.params.len() {
                mask.push(false);
            }
            mask.push(true);
        }

        self.inner.params.push(Param { var, bound });

        var
    }

    pub fn eq(&mut self, a: Var, b: Var) {
        for InstrDefInternal {
            equality: (offset, mask),
            ..
        } in &mut self.inner.instrs
        {
            if mask.is_empty() {
                *offset = self.inner.equality.len();
            }
            while *offset + mask.len() < self.inner.equality.len() {
                mask.push(false);
            }
            mask.push(true);
        }
        self.inner.equality.push((a, b));
    }

    pub fn action(&mut self, action: T, inputs: impl IntoIterator<Item = Var>) -> Var {
        let new_variable = self.variable_builder.next();
        self.action_into(new_variable, action, inputs);
        new_variable
    }

    pub fn action_into(&mut self, dest: Var, action: T, inputs: impl IntoIterator<Item = Var>) {
        let inputs = inputs.into_iter().collect();

        for InstrDefInternal {
            actions: (offset, mask),
            ..
        } in &mut self.inner.instrs
        {
            if mask.is_empty() {
                *offset = self.inner.actions.len();
            }
            while *offset + mask.len() < self.inner.actions.len() {
                mask.push(false);
            }
            mask.push(true);
        }

        self.inner.actions.push(Action {
            dest: dest,
            action: action,
            inputs,
        });
    }

    fn merge(mut self, other: &InstrBuilder<'a, T>) -> Self {
        use std::mem;

        self.inner.params.extend(other.inner.params.iter().cloned());
        self.inner
            .actions
            .extend(other.inner.actions.iter().cloned());
        self.inner
            .equality
            .extend(other.inner.equality.iter().cloned());

        let out = mem::replace(&mut self.inner.instrs, vec![])
            .into_iter()
            .flat_map(|instr| {
                other.inner.instrs.iter().map(move |other| {
                    let mut instr = instr.clone();

                    for _ in 0..other.params.0 {
                        instr.params.1.push(false);
                    }
                    instr.params.1.extend(other.params.1.iter());

                    for _ in 0..other.actions.0 {
                        instr.actions.1.push(false);
                    }
                    instr.actions.1.extend(other.actions.1.iter());

                    for _ in 0..other.equality.0 {
                        instr.equality.1.push(false);
                    }
                    instr.equality.1.extend(other.equality.1.iter());

                    instr
                })
            });

        self.inner.instrs = out.collect();

        self
    }

    pub fn variants<C: ArrayLength<Var>>(&mut self) -> Variants<'_, 'a, GenericArray<Var, C>, T> {
        let vars = GenericArray::generate(|_| self.variable_builder.next());

        Variants {
            variables: vars,
            variable_builder: self.variable_builder.clone(),
            variant_builders: vec![],
            builder: self,
        }
    }
}

pub struct Variants<'borrow, 'a, V, T> {
    builder: &'borrow mut InstrBuilder<'a, T>,
    variable_builder: VariableBuilder,
    variant_builders: Vec<InstrBuilder<'a, T>>,
    variables: V,
}

impl<V, T> Variants<'_, '_, V, T>
where
    V: Clone + IntoArray,
    T: Clone,
{
    pub fn or<F>(mut self, func: F) -> Self
    where
        F: FnOnce(V::Array, &mut InstrBuilder<T>),
    {
        let mut builder = InstrBuilder::new(self.builder.name);
        builder.variable_builder = self.variable_builder.clone();

        func(self.variables.clone().into_array(), &mut builder);

        self.builder.variable_builder.cur = self
            .builder
            .variable_builder
            .cur
            .max(builder.variable_builder.cur);

        self.variant_builders.push(builder);

        self
    }
}

impl<V, T> Variants<'_, '_, V, T>
where
    V: IntoArray,
    T: Clone,
{
    pub fn finish(mut self) -> V::Array {
        self.builder.inner = std::mem::replace(&mut self.variant_builders, vec![])
            .into_iter()
            .map(|build| build.merge(self.builder).inner)
            .fold(MachineSpec::new(), |last, cur| last.merge(cur));

        self.variables.into_array()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Action<T> {
    pub dest: Var,
    pub action: T,
    pub inputs: Vec<Var>,
}
