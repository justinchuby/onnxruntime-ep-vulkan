//! Dtype sets — the `caps` column of a registry row.
//!
//! One op's dtype policy has to be readable in the row itself (so that adding an op is adding a
//! row), *and* it has to drive the build-time shader variant matrix (so we generate exactly the
//! variants we claim and no more). A small bitset over [`DType`] does both.

use crate::engine::DType;
use std::fmt;

/// Every dtype the engine has storage for, in a fixed order that indexes [`DTypeSet`]'s bits.
pub const ALL_DTYPES: [DType; 6] = [
    DType::F32,
    DType::F16,
    DType::I64,
    DType::I32,
    DType::U8,
    DType::Bool,
];

/// Number of dtypes; the width of the variant table.
pub const DTYPE_COUNT: usize = ALL_DTYPES.len();

/// The dtype's index in [`ALL_DTYPES`], which is also its bit position in a [`DTypeSet`] and its
/// column in the shader variant table.
pub const fn dtype_index(d: DType) -> usize {
    match d {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::I64 => 2,
        DType::I32 => 3,
        DType::U8 => 4,
        DType::Bool => 5,
    }
}

/// The dtype's shader-variant suffix, e.g. `f32` in `ew_binary_add_f32`.
///
/// Also the spelling used in decline messages, so a user reading `[dtype] ... supports f32, f16`
/// and a user reading a shader stem see the same words.
pub const fn dtype_suffix(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::I64 => "i64",
        DType::I32 => "i32",
        DType::U8 => "u8",
        DType::Bool => "bool",
    }
}

/// A set of [`DType`]s, cheap enough to sit in a `const` table.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DTypeSet(u8);

impl DTypeSet {
    /// The empty set. No row may declare it; the registry tests enforce that.
    pub const EMPTY: DTypeSet = DTypeSet(0);

    /// Build a set from a slice, usable in `const` context.
    pub const fn of(items: &[DType]) -> DTypeSet {
        let mut bits = 0u8;
        let mut i = 0;
        while i < items.len() {
            bits |= 1u8 << dtype_index(items[i]);
            i += 1;
        }
        DTypeSet(bits)
    }

    /// Union, usable in `const` context (there is no `const` `BitOr`).
    pub const fn union(self, other: DTypeSet) -> DTypeSet {
        DTypeSet(self.0 | other.0)
    }

    /// Set difference, for rows that want "the float set, but not f16".
    pub const fn without(self, other: DTypeSet) -> DTypeSet {
        DTypeSet(self.0 & !other.0)
    }

    /// Membership.
    pub const fn contains(self, d: DType) -> bool {
        self.0 & (1u8 << dtype_index(d)) != 0
    }

    /// True when no dtype is in the set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// How many dtypes are in the set — i.e. how many shader variants this row generates.
    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// The dtypes in [`ALL_DTYPES`] order.
    pub fn iter(self) -> impl Iterator<Item = DType> {
        ALL_DTYPES.into_iter().filter(move |d| self.contains(*d))
    }

    /// The raw bits, for tests and for stable serialisation of a capability report.
    pub const fn bits(self) -> u8 {
        self.0
    }
}

impl fmt::Display for DTypeSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("(none)");
        }
        let mut first = true;
        for d in self.iter() {
            if !first {
                f.write_str(", ")?;
            }
            f.write_str(dtype_suffix(d))?;
            first = false;
        }
        Ok(())
    }
}

impl fmt::Debug for DTypeSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DTypeSet({self})")
    }
}

// -----------------------------------------------------------------------------------------
// Named sets — the vocabulary the op tables are written in
// -----------------------------------------------------------------------------------------

/// `f32` only. The safe floor: every device supports it.
pub const F32: DTypeSet = DTypeSet::of(&[DType::F32]);

/// `f16` only.
pub const F16: DTypeSet = DTypeSet::of(&[DType::F16]);

/// Floating point. `f16` variants are generated but a device without `shaderFloat16` +
/// `storageBuffer16BitAccess` will decline them at claim time (OQ-M2).
pub const FLOAT: DTypeSet = DTypeSet::of(&[DType::F32, DType::F16]);

/// Signed integers we index and count with.
pub const INT: DTypeSet = DTypeSet::of(&[DType::I64, DType::I32]);

/// Everything arithmetic: the default for `Add`-shaped ops.
pub const NUMERIC: DTypeSet = DTypeSet::of(&[DType::F32, DType::F16, DType::I64, DType::I32]);

/// `bool` only — the logical ops.
pub const BOOL: DTypeSet = DTypeSet::of(&[DType::Bool]);

/// Every dtype the engine has storage for. Used by `Where`, `Cast` and the shape ops, which are
/// dtype-agnostic up to element size.
pub const ANY: DTypeSet = DTypeSet::of(&ALL_DTYPES);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_are_unique_and_dense() {
        let mut seen = [false; DTYPE_COUNT];
        for d in ALL_DTYPES {
            let i = dtype_index(d);
            assert!(i < DTYPE_COUNT);
            assert!(!seen[i], "duplicate index for {d:?}");
            seen[i] = true;
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn suffixes_are_unique() {
        let mut s: Vec<&str> = ALL_DTYPES.iter().map(|d| dtype_suffix(*d)).collect();
        s.sort_unstable();
        let before = s.len();
        s.dedup();
        assert_eq!(before, s.len(), "shader variant suffixes must be unique");
    }

    #[test]
    fn membership_matches_construction() {
        assert!(FLOAT.contains(DType::F32));
        assert!(FLOAT.contains(DType::F16));
        assert!(!FLOAT.contains(DType::I32));
        assert!(!FLOAT.contains(DType::Bool));
        assert_eq!(FLOAT.len(), 2);
    }

    #[test]
    fn empty_set_is_empty() {
        assert!(DTypeSet::EMPTY.is_empty());
        assert_eq!(DTypeSet::EMPTY.len(), 0);
        assert_eq!(DTypeSet::EMPTY.iter().count(), 0);
        assert_eq!(DTypeSet::EMPTY.to_string(), "(none)");
    }

    #[test]
    fn any_contains_everything() {
        assert_eq!(ANY.len(), DTYPE_COUNT);
        for d in ALL_DTYPES {
            assert!(ANY.contains(d));
        }
    }

    #[test]
    fn union_and_without_are_const_correct() {
        const BOTH: DTypeSet = FLOAT.union(INT);
        assert_eq!(BOTH, NUMERIC);
        const F32_ONLY: DTypeSet = FLOAT.without(F16);
        assert_eq!(F32_ONLY, F32);
    }

    #[test]
    fn iteration_is_in_declaration_order() {
        let seen: Vec<DType> = NUMERIC.iter().collect();
        assert_eq!(seen, vec![DType::F32, DType::F16, DType::I64, DType::I32]);
    }

    #[test]
    fn display_lists_suffixes() {
        assert_eq!(FLOAT.to_string(), "f32, f16");
        assert_eq!(BOOL.to_string(), "bool");
    }
}
