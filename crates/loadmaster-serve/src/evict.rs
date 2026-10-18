//! Footprint-based eviction (spec task 2.4), as a pure function.

/// A model currently holding memory (loaded or loading).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resident {
    pub name: String,
    pub bytes: u64,
    /// Larger is more recent.
    pub last_used: u64,
    pub keep_warm: bool,
    /// Serving requests now; never evicted.
    pub busy: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Unload these, in order, then load. Empty when the model already fits.
    Load { evict: Vec<String> },
    /// Even evicting everything evictable leaves too little.
    CannotFit { need: u64, available: u64 },
}

/// Decide what to unload so `incoming` bytes fit in `budget`.
///
/// Least recently used first; `keep_warm` models only when the others do not free
/// enough; busy models never. Evicts exactly as many as needed.
pub fn schedule(resident: &[Resident], incoming: u64, budget: u64) -> Decision {
    let used: u64 = resident.iter().map(|r| r.bytes).sum();
    if used + incoming <= budget {
        return Decision::Load { evict: vec![] };
    }
    let mut order: Vec<&Resident> = resident.iter().filter(|r| !r.busy).collect();
    order.sort_by_key(|r| (r.keep_warm, r.last_used, r.name.clone()));
    let mut freed = 0;
    let mut evict = Vec::new();
    for r in order {
        if used - freed + incoming <= budget {
            break;
        }
        freed += r.bytes;
        evict.push(r.name.clone());
    }
    if used - freed + incoming <= budget {
        Decision::Load { evict }
    } else {
        Decision::CannotFit {
            need: incoming,
            available: budget.saturating_sub(used - freed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn r(name: &str, bytes: u64, last_used: u64, keep_warm: bool, busy: bool) -> Resident {
        Resident {
            name: name.into(),
            bytes,
            last_used,
            keep_warm,
            busy,
        }
    }

    #[test]
    fn evicts_lru_first_and_only_as_needed() {
        let res = [
            r("a", 8, 1, false, false),
            r("b", 8, 3, false, false),
            r("c", 8, 2, false, false),
        ];
        assert_eq!(
            schedule(&res, 8, 24),
            Decision::Load {
                evict: vec!["a".into()]
            }
        );
        assert_eq!(
            schedule(&res, 16, 24),
            Decision::Load {
                evict: vec!["a".into(), "c".into()]
            }
        );
        assert_eq!(schedule(&res, 0, 24), Decision::Load { evict: vec![] });
    }

    #[test]
    fn keep_warm_goes_last_and_busy_never() {
        let res = [
            r("warm", 10, 0, true, false),
            r("cold", 10, 5, false, false),
        ];
        assert_eq!(
            schedule(&res, 10, 20),
            Decision::Load {
                evict: vec!["cold".into()]
            }
        );
        assert_eq!(
            schedule(&res, 20, 20),
            Decision::Load {
                evict: vec!["cold".into(), "warm".into()]
            }
        );
        let res = [r("busy", 10, 0, false, true)];
        assert_eq!(
            schedule(&res, 15, 20),
            Decision::CannotFit {
                need: 15,
                available: 10
            }
        );
        assert!(matches!(schedule(&[], 30, 20), Decision::CannotFit { .. }));
    }

    proptest! {
        #[test]
        fn result_fits_or_cannot_fit(
            res in prop::collection::vec((1u64..40, 0u64..100, any::<bool>(), any::<bool>()), 0..8),
            incoming in 0u64..60,
            budget in 0u64..120,
        ) {
            let res: Vec<Resident> = res.iter().enumerate()
                .map(|(i, &(b, t, w, busy))| r(&format!("m{i}"), b, t, w, busy)).collect();
            match schedule(&res, incoming, budget) {
                Decision::Load { evict } => {
                    let left: u64 = res.iter().filter(|x| !evict.contains(&x.name)).map(|x| x.bytes).sum();
                    prop_assert!(left + incoming <= budget);
                    prop_assert!(evict.iter().all(|e| !res.iter().find(|x| &x.name == e).unwrap().busy));
                    // Minimal: keeping the last evicted model would not fit.
                    if let Some(last) = evict.last() {
                        let b = res.iter().find(|x| &x.name == last).unwrap().bytes;
                        prop_assert!(left + b + incoming > budget);
                    }
                    // keep_warm only when every non-busy cold model is already gone.
                    if evict.iter().any(|e| res.iter().find(|x| &x.name == e).unwrap().keep_warm) {
                        prop_assert!(res.iter().filter(|x| !x.keep_warm && !x.busy).all(|x| evict.contains(&x.name)));
                    }
                }
                Decision::CannotFit { .. } => {
                    let pinned: u64 = res.iter().filter(|x| x.busy).map(|x| x.bytes).sum();
                    prop_assert!(pinned + incoming > budget);
                }
            }
        }
    }
}
