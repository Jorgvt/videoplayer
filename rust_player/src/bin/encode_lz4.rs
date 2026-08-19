use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use rayon::prelude::*;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const FRAME_SIZE: usize = WIDTH * HEIGHT * 3; // RGB24 frame size = 2,764,800 bytes
const MAGIC: &[u8; 4] = b"GLZ4";
const VERSION: u32 = 1;

fn encode_rgb_file_to_lz4(input_path: &Path, output_path: &Path) {
    println!("-----------------------------------------------------------------");
    println!("Encoding: {:?}", input_path.file_name().unwrap());
    println!("Source  : {:?}", input_path);
    println!("Target  : {:?}", output_path);

    let t_start = Instant::now();
    let raw_data = fs::read(input_path).expect("Failed to read input .rgb file");
    let num_frames = raw_data.len() / FRAME_SIZE;
    println!("Read {:.2} MB ({} frames) in {:.3}s", raw_data.len() as f64 / 1024.0 / 1024.0, num_frames, t_start.elapsed().as_secs_f64());

    let t_compress_start = Instant::now();

    // Compress each frame in parallel using Rayon
    let frame_slices: Vec<&[u8]> = (0..num_frames)
        .map(|i| &raw_data[i * FRAME_SIZE..(i + 1) * FRAME_SIZE])
        .collect();

    let compressed_frames: Vec<Vec<u8>> = frame_slices
        .par_iter()
        .map(|frame| {
            let compressed = lz4_flex::compress_prepend_size(frame);
            // Verify bit-for-bit lossless parity
            let decompressed = lz4_flex::decompress_size_prepended(&compressed).expect("Decompression failed");
            assert_eq!(*frame, &decompressed[..], "Bit-exact lossless check failed!");
            compressed
        })
        .collect();

    let t_compress = t_compress_start.elapsed().as_secs_f64();

    // Write binary container
    let out_file = File::create(output_path).expect("Failed to create output file");
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, out_file);

    // Header (24 bytes)
    writer.write_all(MAGIC).unwrap();
    writer.write_all(&VERSION.to_le_bytes()).unwrap();
    writer.write_all(&(WIDTH as u32).to_le_bytes()).unwrap();
    writer.write_all(&(HEIGHT as u32).to_le_bytes()).unwrap();
    writer.write_all(&(num_frames as u32).to_le_bytes()).unwrap();
    writer.write_all(&(FRAME_SIZE as u32).to_le_bytes()).unwrap();

    // Frame size table
    for f in &compressed_frames {
        writer.write_all(&(f.len() as u32).to_le_bytes()).unwrap();
    }

    // Contiguous compressed payload
    let mut total_compressed_bytes = 0usize;
    for f in &compressed_frames {
        writer.write_all(f).unwrap();
        total_compressed_bytes += f.len();
    }
    writer.flush().unwrap();

    let original_mb = raw_data.len() as f64 / 1024.0 / 1024.0;
    let compressed_mb = (24 + num_frames * 4 + total_compressed_bytes) as f64 / 1024.0 / 1024.0;
    let ratio = (compressed_mb / original_mb) * 100.0;
    let encode_throughput_mb_s = original_mb / t_compress;

    println!("Compressed in {:.3}s ({:.1} MB/s)", t_compress, encode_throughput_mb_s);
    println!("Original Size  : {:.2} MB", original_mb);
    println!("Compressed Size: {:.2} MB ({:.1}% of original, {:.2}x compression)", compressed_mb, ratio, original_mb / compressed_mb);
    println!("Lossless Check : 100% BIT-FOR-BIT IDENTICAL (Verified across all {} frames)", num_frames);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let raw_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("/home/jv495/Downloads/GAIM240_refs_raw")
    };

    let lz4_dir = if args.len() > 2 {
        PathBuf::from(&args[2])
    } else {
        PathBuf::from("/home/jv495/Downloads/GAIM240_refs_lz4")
    };

    fs::create_dir_all(&lz4_dir).expect("Failed to create LZ4 directory");

    println!("=================================================================");
    println!("  GAIM240 LOSSLESS LZ4 FRAME CONVERTER & VERIFIER");
    println!("=================================================================");
    println!("Source Raw Dir : {:?}", raw_dir);
    println!("Target LZ4 Dir : {:?}", lz4_dir);

    let mut rgb_files = Vec::new();
    if let Ok(entries) = fs::read_dir(&raw_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().map_or(false, |e| e == "rgb") {
                rgb_files.push(p);
            }
        }
    }
    rgb_files.sort();

    if rgb_files.is_empty() {
        println!("Error: No .rgb files found in {:?}", raw_dir);
        return;
    }

    println!("Found {} .rgb files to convert.\n", rgb_files.len());

    for rgb_path in rgb_files {
        let stem = rgb_path.file_stem().unwrap().to_string_lossy();
        let out_path = lz4_dir.join(format!("{}.lz4", stem));
        encode_rgb_file_to_lz4(&rgb_path, &out_path);
    }

    println!("\n=================================================================");
    println!("  ALL SCENES CONVERTED AND 100% VERIFIED LOSSLESS");
    println!("=================================================================");
}
