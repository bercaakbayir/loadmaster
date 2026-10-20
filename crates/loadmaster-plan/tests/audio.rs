//! Audio projectors: the encoder is counted under its own name, sized for one 30 s clip, and
//! moves to the CPU with the rest of the projector.

mod common;

use common::*;
use loadmaster_hw::FakeHw;
use loadmaster_plan::{PlanRequest, explain, plan};

fn at(ctx: u32, mmproj_offload: bool) -> PlanRequest {
    PlanRequest {
        ctx: Some(ctx),
        mmproj_offload,
        ..PlanRequest::default()
    }
}

#[test]
fn an_audio_projector_is_planned_for_one_30_s_clip() {
    let hw = FakeHw::rtx_3060_12g().0;
    let m = model(ULTRAVOX_8B);
    let mm = m.mmproj.as_ref().expect("the projector beside the model");
    assert!(mm.vision.is_none());
    let audio = mm.audio.as_ref().unwrap();
    assert_eq!(
        (audio.projector.as_str(), audio.n_tokens),
        ("ultravox", 187)
    );
    assert_eq!(m.kind_label(), "audio");

    let p = plan(&m, &hw, &hw.gpus[0], &at(8192, true)).unwrap();
    assert!(p.footprint.term("audio_compute").is_some());
    assert!(p.footprint.term("vision_compute").is_none());
    assert!(
        p.footprint
            .term("mmproj")
            .unwrap()
            .detail
            .contains("one 30 s clip: 187 tokens")
    );
    let text = explain::summary(&p, explain::Style::PLAIN);
    assert!(text.contains("on GPU, one 30 s clip"), "{text}");

    let cpu = plan(&m, &hw, &hw.gpus[0], &at(8192, false)).unwrap();
    assert!(cpu.argv.contains(&"--no-mmproj-offload".to_owned()));
    assert!(cpu.footprint.term("audio_compute").is_none());
    let host: Vec<&str> = cpu.footprint.host.iter().map(|t| t.key).collect();
    assert!(
        host.contains(&"mmproj_cpu") && host.contains(&"audio_compute_cpu"),
        "{host:?}"
    );
}

#[test]
fn the_projector_counts_as_part_of_the_vram_budget() {
    let hw = FakeHw::rtx_3060_12g().0;
    let with = plan(&model(ULTRAVOX_8B), &hw, &hw.gpus[0], &at(8192, true)).unwrap();
    let without = plan(&model(LLAMA_8B), &hw, &hw.gpus[0], &at(8192, true)).unwrap();
    let mm = with.footprint.term("mmproj").unwrap().bytes
        + with.footprint.term("audio_compute").unwrap().bytes;
    assert_eq!(with.footprint.gpu_bytes, without.footprint.gpu_bytes + mm);
}
