//! ggml tensor types and llama.cpp file types (`general.file_type`).

use serde::Serialize;

/// A ggml tensor element type. Discriminants match `enum ggml_type` in ggml.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            16 => IQ2_XXS,
            17 => IQ2_XS,
            18 => IQ3_XXS,
            19 => IQ1_S,
            20 => IQ4_NL,
            21 => IQ3_S,
            22 => IQ2_S,
            23 => IQ4_XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => IQ1_M,
            30 => BF16,
            34 => TQ1_0,
            35 => TQ2_0,
            39 => MXFP4,
            _ => return None,
        })
    }

    pub fn id(self) -> u32 {
        use GgmlType::*;
        match self {
            F32 => 0,
            F16 => 1,
            Q4_0 => 2,
            Q4_1 => 3,
            Q5_0 => 6,
            Q5_1 => 7,
            Q8_0 => 8,
            Q8_1 => 9,
            Q2_K => 10,
            Q3_K => 11,
            Q4_K => 12,
            Q5_K => 13,
            Q6_K => 14,
            Q8_K => 15,
            IQ2_XXS => 16,
            IQ2_XS => 17,
            IQ3_XXS => 18,
            IQ1_S => 19,
            IQ4_NL => 20,
            IQ3_S => 21,
            IQ2_S => 22,
            IQ4_XS => 23,
            I8 => 24,
            I16 => 25,
            I32 => 26,
            I64 => 27,
            F64 => 28,
            IQ1_M => 29,
            BF16 => 30,
            TQ1_0 => 34,
            TQ2_0 => 35,
            MXFP4 => 39,
        }
    }

    /// `(elements per block, bytes per block)`, from the `block_*` structs in ggml-common.h.
    pub fn block(self) -> (u64, u64) {
        use GgmlType::*;
        match self {
            F32 | I32 => (1, 4),
            F16 | BF16 | I16 => (1, 2),
            I8 => (1, 1),
            I64 | F64 => (1, 8),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q8_1 => (32, 36),
            IQ4_NL => (32, 18),
            MXFP4 => (32, 17),
            Q2_K => (256, 84),
            Q3_K => (256, 110),
            Q4_K => (256, 144),
            Q5_K => (256, 176),
            Q6_K => (256, 210),
            Q8_K => (256, 292),
            IQ2_XXS => (256, 66),
            IQ2_XS => (256, 74),
            IQ3_XXS => (256, 98),
            IQ1_S => (256, 50),
            IQ3_S => (256, 110),
            IQ2_S => (256, 82),
            IQ4_XS => (256, 136),
            IQ1_M => (256, 56),
            TQ1_0 => (256, 54),
            TQ2_0 => (256, 66),
        }
    }

    /// Bytes for a row of `n` elements, as `ggml_row_size`. `None` if `n` is not a whole
    /// number of blocks, which ggml rejects.
    pub fn row_bytes(self, n: u64) -> Option<u64> {
        let (blk, size) = self.block();
        (n % blk == 0).then(|| n / blk * size)
    }

    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "f32",
            F16 => "f16",
            Q4_0 => "q4_0",
            Q4_1 => "q4_1",
            Q5_0 => "q5_0",
            Q5_1 => "q5_1",
            Q8_0 => "q8_0",
            Q8_1 => "q8_1",
            Q2_K => "q2_K",
            Q3_K => "q3_K",
            Q4_K => "q4_K",
            Q5_K => "q5_K",
            Q6_K => "q6_K",
            Q8_K => "q8_K",
            IQ2_XXS => "iq2_xxs",
            IQ2_XS => "iq2_xs",
            IQ3_XXS => "iq3_xxs",
            IQ1_S => "iq1_s",
            IQ4_NL => "iq4_nl",
            IQ3_S => "iq3_s",
            IQ2_S => "iq2_s",
            IQ4_XS => "iq4_xs",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            F64 => "f64",
            IQ1_M => "iq1_m",
            BF16 => "bf16",
            TQ1_0 => "tq1_0",
            TQ2_0 => "tq2_0",
            MXFP4 => "mxfp4",
        }
    }
}

/// Name of a `general.file_type` value (`enum llama_ftype` in llama.h).
pub fn file_type_name(ftype: u32) -> Option<&'static str> {
    Some(match ftype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip() {
        for id in 0..64 {
            if let Some(t) = GgmlType::from_id(id) {
                assert_eq!(t.id(), id, "{t:?}");
            }
        }
    }

    #[test]
    fn row_bytes_matches_ggml() {
        // 4096-wide rows, as in a Llama-3-8B projection.
        assert_eq!(GgmlType::F16.row_bytes(4096), Some(8192));
        assert_eq!(GgmlType::Q8_0.row_bytes(4096), Some(4352));
        assert_eq!(GgmlType::Q4_K.row_bytes(4096), Some(2304));
        assert_eq!(GgmlType::Q6_K.row_bytes(4096), Some(3360));
        assert_eq!(GgmlType::Q4_K.row_bytes(100), None);
    }
}
