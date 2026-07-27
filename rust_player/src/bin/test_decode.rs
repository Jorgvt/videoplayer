use std::path::Path;
use std::process::Command;
use std::time::Instant;

fn decode_test_opt(path: &str, name: &str, extra_flags: &[&str]) {
    let t0 = Instant::now();
    let ffmpeg_bin = if Path::new("rust_player/lib/usr/bin/ffmpeg").exists() {
        "rust_player/lib/usr/bin/ffmpeg"
    } else if Path::new("lib/usr/bin/ffmpeg").exists() {
        "lib/usr/bin/ffmpeg"
    } else {
        "ffmpeg"
    };

    let mut cmd = Command::new(ffmpeg_bin);
    cmd.env(
        "LD_LIBRARY_PATH",
        "rust_player/lib/usr/lib/x86_64-linux-gnu:lib/usr/lib/x86_64-linux-gnu",
    );

    cmd.args(extra_flags);
    cmd.args([
        "-i",
        path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "pipe:1",
    ]);

    let output = cmd.output().expect("Failed to execute ffmpeg");
    let elapsed = t0.elapsed().as_secs_f64();
    let num_bytes = output.stdout.len();
    let frame_size = 1280 * 720 * 3 / 2;
    let num_frames = num_bytes / frame_size;
    let fps = if elapsed > 0.0 { num_frames as f64 / elapsed } else { 0.0 };

    println!(
        "{:<30} | Decoded {:4} frames ({:.2} MB) in {:.3} s | Speed = {:.2} FPS ({:.2}x)",
        name,
        num_frames,
        num_bytes as f64 / (1024.0 * 1024.0),
        elapsed,
        fps,
        fps / 240.0
    );
}

fn main() {
    let test_file = "/home/jv495/Datasets/GAIM240/zeroday_restir_level2.mp4";
    println!("=== Testing FFmpeg Subprocess Parameter Optimizations ===");
    println!("File: {}\n", test_file);

    decode_test_opt(test_file, "1. Standard Baseline", &[]);
    decode_test_opt(
        test_file,
        "2. Fast Probe (-probesize 32)",
        &["-probesize", "32", "-analyzeduration", "0"],
    );
    decode_test_opt(
        test_file,
        "3. Fast Probe + Threads 8",
        &["-probesize", "32", "-analyzeduration", "0", "-threads", "8"],
    );
    decode_test_opt(
        test_file,
        "4. Fast Flags (-flags2 +fast)",
        &[
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-flags2",
            "+fast",
            "-threads",
            "8",
        ],
    );
}
