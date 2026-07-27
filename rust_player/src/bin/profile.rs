use std::env;
use std::ffi::CString;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use glfw::{Action, Context, Key, WindowHint};

#[derive(Debug, Clone)]
struct FrameData {
    width: i32,
    height: i32,
    data: Vec<u8>,
}

fn get_dataset_dir() -> PathBuf {
    let default_path = PathBuf::from("/home/jv495/Datasets/GAIM240");
    if default_path.exists() {
        default_path
    } else {
        PathBuf::from("GAIM240")
    }
}

fn decode_video_cmd(path: &Path) -> Arc<Vec<FrameData>> {
    use std::process::Command;
    let ffmpeg_bin = if Path::new("rust_player/lib/usr/bin/ffmpeg").exists() {
        "rust_player/lib/usr/bin/ffmpeg"
    } else if Path::new("lib/usr/bin/ffmpeg").exists() {
        "lib/usr/bin/ffmpeg"
    } else {
        "ffmpeg"
    };

    let path_str = path.to_string_lossy();
    let output = Command::new(ffmpeg_bin)
        .env(
            "LD_LIBRARY_PATH",
            "rust_player/lib/usr/lib/x86_64-linux-gnu:lib/usr/lib/x86_64-linux-gnu",
        )
        .args([
            "-hwaccel",
            "cuda",
            "-i",
            &path_str,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ])
        .output();

    let mut frames_vec = Vec::new();
    if let Ok(out) = output {
        let raw = out.stdout;
        let frame_size = 1280 * 720 * 3;
        let num_frames = raw.len() / frame_size;

        for i in 0..num_frames {
            let start = i * frame_size;
            let end = start + frame_size;
            if end <= raw.len() {
                frames_vec.push(FrameData {
                    width: 1280,
                    height: 720,
                    data: raw[start..end].to_vec(),
                });
            }
        }
    }

    if frames_vec.is_empty() {
        // CPU fallback if NVDEC / CUDA flags failed
        let output_cpu = Command::new(ffmpeg_bin)
            .args([
                "-i",
                &path_str,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ])
            .output();
        if let Ok(out) = output_cpu {
            let raw = out.stdout;
            let frame_size = 1280 * 720 * 3;
            let num_frames = raw.len() / frame_size;
            for i in 0..num_frames {
                let start = i * frame_size;
                let end = start + frame_size;
                if end <= raw.len() {
                    frames_vec.push(FrameData {
                        width: 1280,
                        height: 720,
                        data: raw[start..end].to_vec(),
                    });
                }
            }
        }
    }

    Arc::new(frames_vec)
}

struct QuadShader {
    program: u32,
    vbo: u32,
}

impl QuadShader {
    fn new() -> Self {
        let vert_code = CString::new(
            "
            #version 120
            attribute vec2 position;
            attribute vec2 texcoord;
            varying vec2 v_texcoord;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_texcoord = texcoord;
            }
        ",
        )
        .unwrap();

        let frag_code = CString::new(
            "
            #version 120
            uniform sampler2D u_texture;
            varying vec2 v_texcoord;
            void main() {
                gl_FragColor = texture2D(u_texture, v_texcoord);
            }
        ",
        )
        .unwrap();

        unsafe {
            let vert_shader = gl::CreateShader(gl::VERTEX_SHADER);
            gl::ShaderSource(vert_shader, 1, &vert_code.as_ptr(), std::ptr::null());
            gl::CompileShader(vert_shader);

            let frag_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
            gl::ShaderSource(frag_shader, 1, &frag_code.as_ptr(), std::ptr::null());
            gl::CompileShader(frag_shader);

            let program = gl::CreateProgram();
            gl::AttachShader(program, vert_shader);
            gl::AttachShader(program, frag_shader);
            gl::LinkProgram(program);

            gl::DeleteShader(vert_shader);
            gl::DeleteShader(frag_shader);

            let mut vbo = 0;
            gl::GenBuffers(1, &mut vbo);

            QuadShader { program, vbo }
        }
    }

    fn draw_quad(&self, x1: f32, y1: f32, x2: f32, y2: f32, tex_id: u32) {
        #[repr(C)]
        struct Vertex {
            pos: [f32; 2],
            uv: [f32; 2],
        }

        let vertices: [Vertex; 6] = [
            Vertex { pos: [x1, y1], uv: [0.0, 1.0] },
            Vertex { pos: [x2, y1], uv: [1.0, 1.0] },
            Vertex { pos: [x2, y2], uv: [1.0, 0.0] },
            Vertex { pos: [x1, y1], uv: [0.0, 1.0] },
            Vertex { pos: [x2, y2], uv: [1.0, 0.0] },
            Vertex { pos: [x1, y2], uv: [0.0, 0.0] },
        ];

        unsafe {
            gl::UseProgram(self.program);
            gl::BindTexture(gl::TEXTURE_2D, tex_id);

            gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertices.len() * std::mem::size_of::<Vertex>()) as isize,
                vertices.as_ptr() as *const _,
                gl::DYNAMIC_DRAW,
            );

            let pos_attr = CString::new("position").unwrap();
            let pos_loc = gl::GetAttribLocation(self.program, pos_attr.as_ptr());
            gl::EnableVertexAttribArray(pos_loc as u32);
            gl::VertexAttribPointer(
                pos_loc as u32,
                2,
                gl::FLOAT,
                gl::FALSE,
                std::mem::size_of::<Vertex>() as i32,
                0 as *const _,
            );

            let uv_attr = CString::new("texcoord").unwrap();
            let uv_loc = gl::GetAttribLocation(self.program, uv_attr.as_ptr());
            gl::EnableVertexAttribArray(uv_loc as u32);
            gl::VertexAttribPointer(
                uv_loc as u32,
                2,
                gl::FLOAT,
                gl::FALSE,
                std::mem::size_of::<Vertex>() as i32,
                (2 * std::mem::size_of::<f32>()) as *const _,
            );

            gl::DrawArrays(gl::TRIANGLES, 0, 6);
        }
    }
}

fn create_preallocated_textures() -> (u32, u32, u32) {
    let mut textures = [0u32; 3];
    unsafe {
        gl::Enable(gl::TEXTURE_2D);
        gl::GenTextures(3, textures.as_mut_ptr());
        for &tex in &textures {
            gl::BindTexture(gl::TEXTURE_2D, tex);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGB as i32,
                1280,
                720,
                0,
                gl::BGR,
                gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
        }
    }
    (textures[0], textures[1], textures[2])
}

fn render_pyramid(
    window: &mut glfw::Window,
    shader: &QuadShader,
    tex_a: u32,
    tex_ref: u32,
    tex_c: u32,
    fa: &FrameData,
    fref: &FrameData,
    fc: &FrameData,
) {
    let (w, h) = window.get_framebuffer_size();
    let target_aspect = 16.0 / 9.0;

    let mut w_view = w;
    let mut h_view = (w as f32 / target_aspect) as i32;
    if h_view > h {
        h_view = h;
        w_view = (h as f32 * target_aspect) as i32;
    }
    let x_offset = (w - w_view) / 2;
    let y_offset = (h - h_view) / 2;

    unsafe {
        gl::Viewport(0, 0, w, h);
        gl::ClearColor(0.02, 0.02, 0.03, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);

        gl::Viewport(x_offset, y_offset, w_view, h_view);

        gl::BindTexture(gl::TEXTURE_2D, tex_a);
        gl::TexSubImage2D(
            gl::TEXTURE_2D,
            0,
            0,
            0,
            fa.width,
            fa.height,
            gl::BGR,
            gl::UNSIGNED_BYTE,
            fa.data.as_ptr() as *const _,
        );

        gl::BindTexture(gl::TEXTURE_2D, tex_ref);
        gl::TexSubImage2D(
            gl::TEXTURE_2D,
            0,
            0,
            0,
            fref.width,
            fref.height,
            gl::BGR,
            gl::UNSIGNED_BYTE,
            fref.data.as_ptr() as *const _,
        );

        gl::BindTexture(gl::TEXTURE_2D, tex_c);
        gl::TexSubImage2D(
            gl::TEXTURE_2D,
            0,
            0,
            0,
            fc.width,
            fc.height,
            gl::BGR,
            gl::UNSIGNED_BYTE,
            fc.data.as_ptr() as *const _,
        );

        shader.draw_quad(-0.5, 0.0, 0.5, 1.0, tex_ref);
        shader.draw_quad(-1.0, -1.0, 0.0, 0.0, tex_a);
        shader.draw_quad(0.0, -1.0, 1.0, 0.0, tex_c);
    }
}

struct FrameProfileRecord {
    frame_index: usize,
    get_image_ms: f64,
    upload_draw_ms: f64,
    flip_ms: f64,
    total_loop_ms: f64,
    swap_timestamp_sec: f64,
    inter_frame_interval_ms: Option<f64>,
}

fn calc_stats(vals: &[f64]) -> (f64, f64, f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    let std_dev = var.sqrt();
    (mean, std_dev, min, max)
}

fn find_scene_videos(
    dataset_dir: &Path,
    scene: &str,
    requested_metric: Option<&str>,
) -> Option<(PathBuf, PathBuf, PathBuf, String)> {
    let ref_path = dataset_dir.join(format!("{}_reference.mp4", scene));
    if !ref_path.exists() {
        println!("Error: Reference video not found at {:?}", ref_path);
        return None;
    }

    let prefix = format!("{}_", scene);
    let mut available_metrics = Vec::new();

    if let Ok(entries) = fs::read_dir(dataset_dir) {
        for entry in entries.flatten() {
            let filename = entry.file_name().to_string_lossy().to_string();
            if filename.starts_with(&prefix) && filename.ends_with(".mp4") {
                let rest = &filename[prefix.len()..filename.len() - 4];
                if rest != "reference" {
                    let metric = if let Some(idx) = rest.find("_level") {
                        &rest[..idx]
                    } else {
                        rest
                    };
                    if !available_metrics.contains(&metric.to_string()) {
                        available_metrics.push(metric.to_string());
                    }
                }
            }
        }
    }

    available_metrics.sort();

    if available_metrics.is_empty() {
        println!("Error: No distortion metrics found for scene '{}'", scene);
        return None;
    }

    let selected_metric = if let Some(req) = requested_metric {
        if available_metrics.iter().any(|m| m == req) {
            req.to_string()
        } else {
            println!("Warning: Requested metric '{}' not found. Available metrics: {:?}", req, available_metrics);
            available_metrics[0].clone()
        }
    } else if available_metrics.iter().any(|m| m == "temporal-resolution-multiplexing") {
        "temporal-resolution-multiplexing".to_string()
    } else {
        available_metrics[0].clone()
    };

    let level1_path = dataset_dir.join(format!("{}_{}_level1.mp4", scene, selected_metric));
    let level2_path = dataset_dir.join(format!("{}_{}_level2.mp4", scene, selected_metric));

    Some((ref_path, level1_path, level2_path, selected_metric))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut no_vsync = false;
    let mut use_pacer = false;
    let mut borderless = false;
    let mut windowed = false;
    let mut target_scene = "marbles".to_string();
    let mut requested_metric: Option<String> = None;
    let mut csv_out_path = "profile_results.csv".to_string();

    for arg in &args[1..] {
        if arg == "--no-vsync" || arg == "--novsync" || arg == "--uncapped" {
            no_vsync = true;
        } else if arg == "--pacer" || arg == "--pace-240" {
            no_vsync = true;
            use_pacer = true;
        } else if arg == "--borderless" {
            borderless = true;
        } else if arg == "--windowed" {
            windowed = true;
        } else if arg.starts_with("--scene=") {
            target_scene = arg.trim_start_matches("--scene=").to_string();
        } else if arg.starts_with("--metric=") {
            requested_metric = Some(arg.trim_start_matches("--metric=").to_string());
        } else if arg.starts_with("--out=") {
            csv_out_path = arg.trim_start_matches("--out=").to_string();
        }
    }

    let dataset_dir = get_dataset_dir();
    println!("=== GAIM240 Rust Video Player Multi-Stage Profiler ===");
    println!("Dataset directory: {:?}", dataset_dir);
    println!("Target scene     : {}", target_scene);
    println!(
        "VSync / Pacer    : {}",
        if use_pacer {
            "SOFTWARE PACER (--pacer 240.000 FPS Locked)"
        } else if no_vsync {
            "DISABLED (--no-vsync Raw FPS Max Throughput)"
        } else {
            "ENABLED (Locked to Hardware Refresh)"
        }
    );
    println!("Output CSV Path  : {}", csv_out_path);

    let (ref_path, level1_path, level2_path, selected_metric) = match find_scene_videos(&dataset_dir, &target_scene, requested_metric.as_deref()) {
        Some(res) => res,
        None => return,
    };

    println!("Selected metric  : {}", selected_metric);
    println!("  Left Video (A) : {:?}", level1_path.file_name().unwrap_or_default());
    println!("  Center (Ref)   : {:?}", ref_path.file_name().unwrap_or_default());
    println!("  Right Video (C): {:?}", level2_path.file_name().unwrap_or_default());

    println!("\n[1/2] Preloading video frames into RAM (CUDA NVDEC / FFMPEG)...");
    let p_a = level1_path.clone();
    let p_ref = ref_path.clone();
    let p_c = level2_path.clone();

    let h_a = thread::spawn(move || decode_video_cmd(&p_a));
    let h_ref = thread::spawn(move || decode_video_cmd(&p_ref));
    let h_c = thread::spawn(move || decode_video_cmd(&p_c));

    let frames_a = h_a.join().unwrap();
    let frames_ref = h_ref.join().unwrap();
    let frames_c = h_c.join().unwrap();

    println!("Successfully loaded frames: Video A={}, Ref={}, Video C={}", frames_a.len(), frames_ref.len(), frames_c.len());

    if frames_a.is_empty() || frames_ref.is_empty() || frames_c.is_empty() {
        println!("Error: Failed to decode frames for one or more video streams.");
        return;
    }

    let total_frames = frames_a.len().min(frames_ref.len()).min(frames_c.len());
    if total_frames == 0 {
        println!("Error: No valid frame sequence decoded.");
        return;
    }

    println!("\n[2/2] Initializing OpenGL Context & Starting Profiling Run...");
    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(WindowHint::Resizable(true));
    glfw.window_hint(WindowHint::DoubleBuffer(true));
    glfw.window_hint(WindowHint::ContextVersion(2, 1));

    let (mon_w, mon_h) = glfw.with_primary_monitor(|_, m| {
        m.and_then(|mon| mon.get_video_mode())
            .map(|mode| (mode.width as u32, mode.height as u32))
            .unwrap_or((1280, 720))
    });

    let (mut window, events) = if borderless {
        glfw.window_hint(WindowHint::Decorated(false));
        let (mut w, ev) = glfw
            .create_window(
                mon_w,
                mon_h,
                "GAIM240 Rust Triple Player Profiler",
                glfw::WindowMode::Windowed,
            )
            .expect("Failed to create borderless window");
        w.set_pos(0, 0);
        (w, ev)
    } else if windowed {
        glfw.create_window(
            1280,
            720,
            "GAIM240 Rust Triple Player Profiler",
            glfw::WindowMode::Windowed,
        )
        .expect("Failed to create windowed GLFW window")
    } else {
        glfw.with_connected_monitors(|glfw_ref, monitors| {
            if let Some(m) = monitors.first() {
                glfw_ref
                    .create_window(
                        mon_w,
                        mon_h,
                        "GAIM240 Rust Triple Player Profiler",
                        glfw::WindowMode::FullScreen(m),
                    )
                    .expect("Failed to create fullscreen window")
            } else {
                glfw_ref
                    .create_window(
                        1280,
                        720,
                        "GAIM240 Rust Triple Player Profiler",
                        glfw::WindowMode::Windowed,
                    )
                    .expect("Failed to create windowed window")
            }
        })
    };

    window.make_current();
    if no_vsync {
        glfw.set_swap_interval(glfw::SwapInterval::None);
    } else {
        glfw.set_swap_interval(glfw::SwapInterval::Sync(1));
    }
    window.set_key_polling(true);

    gl::load_with(|s| window.get_proc_address(s) as *const _);

    let shader = QuadShader::new();
    let (tex_a, tex_ref, tex_c) = create_preallocated_textures();

    let mut records: Vec<FrameProfileRecord> = Vec::with_capacity(total_frames);
    let start_instant = Instant::now();

    println!("\n=== Rust Triple OpenGL Player Profiling Active ===");
    println!("Profiling {} frames...", total_frames);
    println!("Press [Q] or [Esc] to interrupt early.\n");

    let mut step_count = 0;
    let mut prev_swap_sec: Option<f64> = None;

    while !window.should_close() && step_count < total_frames {
        let t0 = Instant::now();

        // 1. Image retrieval from RAM
        let idx_a = step_count % frames_a.len();
        let idx_ref = step_count % frames_ref.len();
        let idx_c = step_count % frames_c.len();

        let fa = &frames_a[idx_a];
        let fref = &frames_ref[idx_ref];
        let fc = &frames_c[idx_c];

        let t1 = Instant::now();

        // 2. Render & submit GPU draw commands
        render_pyramid(&mut window, &shader, tex_a, tex_ref, tex_c, fa, fref, fc);

        let t2 = Instant::now();

        // 3. Swap buffers (VSync wait occurs here if enabled)
        window.swap_buffers();

        let t3 = Instant::now();

        let get_image_ms = (t1 - t0).as_secs_f64() * 1000.0;
        let upload_draw_ms = (t2 - t1).as_secs_f64() * 1000.0;
        let flip_ms = (t3 - t2).as_secs_f64() * 1000.0;
        let total_loop_ms = (t3 - t0).as_secs_f64() * 1000.0;
        let swap_timestamp_sec = (t3 - start_instant).as_secs_f64();

        let inter_frame_interval_ms = prev_swap_sec.map(|prev| (swap_timestamp_sec - prev) * 1000.0);
        prev_swap_sec = Some(swap_timestamp_sec);

        records.push(FrameProfileRecord {
            frame_index: step_count,
            get_image_ms,
            upload_draw_ms,
            flip_ms,
            total_loop_ms,
            swap_timestamp_sec,
            inter_frame_interval_ms,
        });

        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            if let glfw::WindowEvent::Key(Key::Escape | Key::Q, _, Action::Press, _) = event {
                println!("\nProfiling interrupted by user keypress.");
                window.set_should_close(true);
            }
        }

        if use_pacer {
            let target_time = start_instant + std::time::Duration::from_nanos((step_count as u64 + 1) * 4_166_667);
            while Instant::now() < target_time {
                std::hint::spin_loop();
            }
        }

        step_count += 1;
    }

    let elapsed_total = start_instant.elapsed().as_secs_f64();
    let num_records = records.len();

    println!("\n=========================================================");
    println!("              RUST PROFILER SUMMARY RESULTS              ");
    println!("=========================================================");
    println!("Total Frames Profiled: {} frames", num_records);
    println!("Total Elapsed Time   : {:.4} s", elapsed_total);
    if elapsed_total > 0.0 {
        println!("Average Presentation FPS: {:.2} FPS", num_records as f64 / elapsed_total);
    }

    let get_img_vec: Vec<f64> = records.iter().map(|r| r.get_image_ms).collect();
    let up_draw_vec: Vec<f64> = records.iter().map(|r| r.upload_draw_ms).collect();
    let flip_vec: Vec<f64> = records.iter().map(|r| r.flip_ms).collect();
    let total_vec: Vec<f64> = records.iter().map(|r| r.total_loop_ms).collect();
    let inter_vec: Vec<f64> = records.iter().filter_map(|r| r.inter_frame_interval_ms).collect();

    let (s_get_m, s_get_s, s_get_min, s_get_max) = calc_stats(&get_img_vec);
    let (s_up_m, s_up_s, s_up_min, s_up_max) = calc_stats(&up_draw_vec);
    let (s_fl_m, s_fl_s, s_fl_min, s_fl_max) = calc_stats(&flip_vec);
    let (s_tot_m, s_tot_s, s_tot_min, s_tot_max) = calc_stats(&total_vec);
    let (s_int_m, s_int_s, s_int_min, s_int_max) = calc_stats(&inter_vec);

    println!("\nStage Breakdown (mean ± std | min .. max):");
    println!("  1. Fetch Frame (RAM) : {:6.3} ± {:5.3} ms  [{:6.3} .. {:6.3} ms]", s_get_m, s_get_s, s_get_min, s_get_max);
    println!("  2. Upload & Draw GPU: {:6.3} ± {:5.3} ms  [{:6.3} .. {:6.3} ms]", s_up_m, s_up_s, s_up_min, s_up_max);
    println!("  3. VSync Flip Wait  : {:6.3} ± {:5.3} ms  [{:6.3} .. {:6.3} ms]", s_fl_m, s_fl_s, s_fl_min, s_fl_max);
    println!("  ---------------------------------------------------------");
    println!("  Total Loop Cycle   : {:6.3} ± {:5.3} ms  [{:6.3} .. {:6.3} ms]", s_tot_m, s_tot_s, s_tot_min, s_tot_max);
    println!("  Inter-Frame Interval: {:6.3} ± {:5.3} ms  [{:6.3} .. {:6.3} ms]", s_int_m, s_int_s, s_int_min, s_int_max);

    // Save CSV matching Python profiler schema
    let mut file = File::create(&csv_out_path).expect("Failed to create CSV output file");
    writeln!(file, "FrameIndex,GetImageDuration_ms,UploadDrawDuration_ms,FlipDuration_ms,TotalLoopDuration_ms,SwapTimestamp_sec,InterFrameInterval_ms").unwrap();

    for record in &records {
        let inter_str = match record.inter_frame_interval_ms {
            Some(val) => format!("{:.6}", val),
            None => "N/A".to_string(),
        };
        writeln!(
            file,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            record.frame_index,
            record.get_image_ms,
            record.upload_draw_ms,
            record.flip_ms,
            record.total_loop_ms,
            record.swap_timestamp_sec,
            inter_str
        )
        .unwrap();
    }

    println!("\nProfiling CSV saved to: {}", csv_out_path);
    println!("You can now plot this data with Python using: uv run plot_profile.py {}", csv_out_path);
    println!("=========================================================");
}
