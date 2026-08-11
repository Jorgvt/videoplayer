use std::env;
use std::ffi::CString;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::Instant;
use std::io::Read;
use std::process::{Command, Stdio, Child};

use glfw::{Action, Context, Key, WindowHint};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const Y_SIZE: usize = WIDTH * HEIGHT;
const UV_WIDTH: usize = WIDTH / 2;
const UV_HEIGHT: usize = HEIGHT / 2;
const UV_SIZE: usize = UV_WIDTH * UV_HEIGHT;
const FRAME_SIZE: usize = Y_SIZE + UV_SIZE + UV_SIZE; // YUV420P frame size = 1,382,400 bytes

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MasterTrial {
    #[serde(rename = "MasterTrialID")]
    master_trial_id: usize,
    #[serde(rename = "Scene")]
    scene: String,
    #[serde(rename = "ComparisonType")]
    comparison_type: String,
    #[serde(rename = "RefFilename")]
    ref_filename: String,
    #[serde(rename = "RefPath")]
    ref_path: String,
    #[serde(rename = "Vid1_Filename")]
    vid1_filename: String,
    #[serde(rename = "Vid1_Metric")]
    vid1_metric: String,
    #[serde(rename = "Vid1_Level")]
    vid1_level: String,
    #[serde(rename = "Vid1_Path")]
    vid1_path: String,
    #[serde(rename = "Vid2_Filename")]
    vid2_filename: String,
    #[serde(rename = "Vid2_Metric")]
    vid2_metric: String,
    #[serde(rename = "Vid2_Level")]
    vid2_level: String,
    #[serde(rename = "Vid2_Path")]
    vid2_path: String,
}

#[derive(Debug, Clone)]
struct VideoInfo {
    filename: String,
    metric: String,
    level: String,
    path: String,
}

#[derive(Debug, Clone)]
struct PreparedTrial {
    trial_idx: usize,
    master_trial_id: usize,
    scene: String,
    comparison_type: String,
    ref_vid: VideoInfo,
    left_vid: VideoInfo,
    right_vid: VideoInfo,
}

// Pre-allocated VRAM textures for a video stream
struct VramStreamData {
    num_frames: usize,
    textures_y: Vec<u32>,
    textures_u: Vec<u32>,
    textures_v: Vec<u32>,
}

impl Drop for VramStreamData {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteTextures(self.textures_y.len() as i32, self.textures_y.as_ptr());
            gl::DeleteTextures(self.textures_u.len() as i32, self.textures_u.as_ptr());
            gl::DeleteTextures(self.textures_v.len() as i32, self.textures_v.as_ptr());
        }
    }
}

struct VramTrialFrames {
    left_stream: VramStreamData,
    ref_stream: VramStreamData,
    right_stream: VramStreamData,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExperimentResult {
    #[serde(rename = "SubjectID")]
    subject_id: String,
    #[serde(rename = "TrialIndex")]
    trial_index: usize,
    #[serde(rename = "MasterTrialID")]
    master_trial_id: usize,
    #[serde(rename = "TotalTrials")]
    total_trials: usize,
    #[serde(rename = "ComparisonType")]
    comparison_type: String,
    #[serde(rename = "Scene")]
    scene: String,
    #[serde(rename = "LeftMetric")]
    left_metric: String,
    #[serde(rename = "LeftLevel")]
    left_level: String,
    #[serde(rename = "LeftVideo")]
    left_video: String,
    #[serde(rename = "RightMetric")]
    right_metric: String,
    #[serde(rename = "RightLevel")]
    right_level: String,
    #[serde(rename = "RightVideo")]
    right_video: String,
    #[serde(rename = "ChosenSide")]
    chosen_side: String,
    #[serde(rename = "ChosenMetric")]
    chosen_metric: String,
    #[serde(rename = "ChosenLevel")]
    chosen_level: String,
    #[serde(rename = "ChosenVideo")]
    chosen_video: String,
    #[serde(rename = "RejectedMetric")]
    rejected_metric: String,
    #[serde(rename = "RejectedLevel")]
    rejected_level: String,
    #[serde(rename = "RejectedVideo")]
    rejected_video: String,
    #[serde(rename = "ResponseTime_sec")]
    response_time_sec: f64,
    #[serde(rename = "PresentationFPS")]
    presentation_fps: f64,
    #[serde(rename = "PreDecodeTime_sec")]
    pre_decode_time_sec: f64,
}

fn spawn_ffmpeg_cmd(path: &str) -> Child {
    #[cfg(target_os = "windows")]
    let ffmpeg_bin = if std::path::Path::new("rust_player/lib/windows/bin/ffmpeg.exe").exists() {
        "rust_player/lib/windows/bin/ffmpeg.exe"
    } else if std::path::Path::new("lib/windows/bin/ffmpeg.exe").exists() {
        "lib/windows/bin/ffmpeg.exe"
    } else {
        "ffmpeg"
    };

    #[cfg(not(target_os = "windows"))]
    let ffmpeg_bin = if Path::new("rust_player/lib/usr/bin/ffmpeg").exists() {
        "rust_player/lib/usr/bin/ffmpeg"
    } else if Path::new("lib/usr/bin/ffmpeg").exists() {
        "lib/usr/bin/ffmpeg"
    } else {
        "ffmpeg"
    };

    let mut cmd = Command::new(ffmpeg_bin);

    #[cfg(target_os = "linux")]
    cmd.env(
        "LD_LIBRARY_PATH",
        "rust_player/lib/usr/lib/x86_64-linux-gnu:lib/usr/lib/x86_64-linux-gnu",
    );

    #[cfg(target_os = "windows")]
    {
        let dll_dir1 = "rust_player/lib/windows/bin";
        let dll_dir2 = "lib/windows/bin";
        let current_path = std::env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{};{};{}", dll_dir1, dll_dir2, current_path));
    }

    cmd.args([
            "-hwaccel",
            "cuda",
            "-c:v",
            "hevc_cuvid",
            "-i",
            path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn ffmpeg")
}

unsafe fn upload_yuv_frame_data(tex_y: u32, tex_u: u32, tex_v: u32, raw_data: &[u8]) {
    let y_ptr = &raw_data[0];
    let u_ptr = &raw_data[Y_SIZE];
    let v_ptr = &raw_data[Y_SIZE + UV_SIZE];

    // Y Texture
    gl::BindTexture(gl::TEXTURE_2D, tex_y);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
    gl::TexImage2D(
        gl::TEXTURE_2D, 0, gl::RED as i32, WIDTH as i32, HEIGHT as i32, 0,
        gl::RED, gl::UNSIGNED_BYTE, y_ptr as *const _ as *const _
    );

    // U Texture
    gl::BindTexture(gl::TEXTURE_2D, tex_u);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
    gl::TexImage2D(
        gl::TEXTURE_2D, 0, gl::RED as i32, UV_WIDTH as i32, UV_HEIGHT as i32, 0,
        gl::RED, gl::UNSIGNED_BYTE, u_ptr as *const _ as *const _
    );

    // V Texture
    gl::BindTexture(gl::TEXTURE_2D, tex_v);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
    gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
    gl::TexImage2D(
        gl::TEXTURE_2D, 0, gl::RED as i32, UV_WIDTH as i32, UV_HEIGHT as i32, 0,
        gl::RED, gl::UNSIGNED_BYTE, v_ptr as *const _ as *const _
    );
}

struct YuvQuadShader {
    program: u32,
    vbo: u32,
    u_y_loc: i32,
    u_u_loc: i32,
    u_v_loc: i32,
}

impl YuvQuadShader {
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
            uniform sampler2D u_tex_y;
            uniform sampler2D u_tex_u;
            uniform sampler2D u_tex_v;
            varying vec2 v_texcoord;
            void main() {
                float y = texture2D(u_tex_y, v_texcoord).r;
                float u = texture2D(u_tex_u, v_texcoord).r - 0.5;
                float v = texture2D(u_tex_v, v_texcoord).r - 0.5;
                
                float r = y + 1.402 * v;
                float g = y - 0.344136 * u - 0.714136 * v;
                float b = y + 1.772 * u;
                
                gl_FragColor = vec4(r, g, b, 1.0);
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

            let u_y_loc = gl::GetUniformLocation(program, CString::new("u_tex_y").unwrap().as_ptr());
            let u_u_loc = gl::GetUniformLocation(program, CString::new("u_tex_u").unwrap().as_ptr());
            let u_v_loc = gl::GetUniformLocation(program, CString::new("u_tex_v").unwrap().as_ptr());

            YuvQuadShader {
                program,
                vbo,
                u_y_loc,
                u_u_loc,
                u_v_loc,
            }
        }
    }

    fn draw_quad(&self, x1: f32, y1: f32, x2: f32, y2: f32, tex_y: u32, tex_u: u32, tex_v: u32) {
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

            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, tex_y);
            gl::Uniform1i(self.u_y_loc, 0);

            gl::ActiveTexture(gl::TEXTURE1);
            gl::BindTexture(gl::TEXTURE_2D, tex_u);
            gl::Uniform1i(self.u_u_loc, 1);

            gl::ActiveTexture(gl::TEXTURE2);
            gl::BindTexture(gl::TEXTURE_2D, tex_v);
            gl::Uniform1i(self.u_v_loc, 2);

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

fn render_pyramid_subimage_vram(
    window: &mut glfw::Window,
    shader: &YuvQuadShader,
    tf: &VramTrialFrames,
    frame_idx: usize,
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

    let idx_left = if tf.left_stream.num_frames > 0 { frame_idx % tf.left_stream.num_frames } else { 0 };
    let idx_ref = if tf.ref_stream.num_frames > 0 { frame_idx % tf.ref_stream.num_frames } else { 0 };
    let idx_right = if tf.right_stream.num_frames > 0 { frame_idx % tf.right_stream.num_frames } else { 0 };

    unsafe {
        gl::Viewport(0, 0, w, h);
        gl::ClearColor(0.02, 0.02, 0.03, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);

        gl::Viewport(x_offset, y_offset, w_view, h_view);

        shader.draw_quad(
            -0.5, 0.0, 0.5, 1.0,
            tf.ref_stream.textures_y[idx_ref],
            tf.ref_stream.textures_u[idx_ref],
            tf.ref_stream.textures_v[idx_ref]
        );
        shader.draw_quad(
            -1.0, -1.0, 0.0, 0.0,
            tf.left_stream.textures_y[idx_left],
            tf.left_stream.textures_u[idx_left],
            tf.left_stream.textures_v[idx_left]
        );
        shader.draw_quad(
            -0.0, -1.0, 1.0, 0.0,
            tf.right_stream.textures_y[idx_right],
            tf.right_stream.textures_u[idx_right],
            tf.right_stream.textures_v[idx_right]
        );
    }
}

fn get_dataset_dir() -> std::path::PathBuf {
    if let Ok(env_val) = std::env::var("GAIM240_DATASET_DIR") {
        let p = std::path::PathBuf::from(env_val);
        if p.exists() {
            return p;
        }
    }
    for rel_path in &["../../Datasets/GAIM240", "../Datasets/GAIM240", "Datasets/GAIM240"] {
        let p = std::path::PathBuf::from(rel_path);
        if p.exists() {
            return p;
        }
    }
    for fallback in &["D:\\GAIM240", "C:\\Datasets\\GAIM240", "/home/jv495/Datasets/GAIM240", "/home/jv495/Developer/Datasets/GAIM240"] {
        let p = std::path::PathBuf::from(fallback);
        if p.exists() {
            return p;
        }
    }
    std::path::PathBuf::from("GAIM240")
}

fn hash_subject(subject_id: &str) -> u64 {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    subject_id.hash(&mut s);
    s.finish()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let subject_id = "BENCHMARK_PIPELINE".to_string();
    let mut exp_mode = "intra".to_string();
    let mut no_vsync = false;
    let mut use_pacer = false;

    for arg in &args[1..] {
        if arg.starts_with("--mode=") {
            exp_mode = arg.trim_start_matches("--mode=").to_lowercase();
        } else if arg == "--no-vsync" || arg == "--novsync" || arg == "--uncapped" {
            no_vsync = true;
        } else if arg == "--pacer" || arg == "--pace-240" {
            no_vsync = true;
            use_pacer = true;
        }
    }

    let bank_path = if Path::new("all_trials_bank.csv").exists() {
        Path::new("all_trials_bank.csv")
    } else if Path::new("../all_trials_bank.csv").exists() {
        Path::new("../all_trials_bank.csv")
    } else {
        println!("Error: all_trials_bank.csv not found.");
        return;
    };

    let mut rdr = csv::Reader::from_path(bank_path).expect("Failed to open all_trials_bank.csv");
    let master_trials: Vec<MasterTrial> = rdr.deserialize().filter_map(|r| r.ok()).collect();

    let filtered_trials: Vec<MasterTrial> = master_trials
        .into_iter()
        .filter(|t| {
            if exp_mode == "inter" {
                t.comparison_type == "INTER-DISTORTION"
            } else if exp_mode == "intra" {
                t.comparison_type == "INTRA-DISTORTION"
            } else {
                true
            }
        })
        .collect();

    let seed = hash_subject(&subject_id);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut shuffled_master = filtered_trials;
    shuffled_master.shuffle(&mut rng);

    let mut prepared_trials: Vec<PreparedTrial> = Vec::new();
    for (i, row) in shuffled_master.iter().enumerate() {
        let flip: bool = rand::Rng::gen(&mut rng);

        let dataset_dir = get_dataset_dir();
        let left_vid_path = dataset_dir.join(if flip { &row.vid2_filename } else { &row.vid1_filename }).to_string_lossy().to_string();
        let right_vid_path = dataset_dir.join(if flip { &row.vid1_filename } else { &row.vid2_filename }).to_string_lossy().to_string();
        let ref_vid_path = dataset_dir.join(&row.ref_filename).to_string_lossy().to_string();

        let left_vid = if flip {
            VideoInfo {
                filename: row.vid2_filename.clone(),
                metric: row.vid2_metric.clone(),
                level: row.vid2_level.clone(),
                path: left_vid_path,
            }
        } else {
            VideoInfo {
                filename: row.vid1_filename.clone(),
                metric: row.vid1_metric.clone(),
                level: row.vid1_level.clone(),
                path: left_vid_path,
            }
        };

        let right_vid = if flip {
            VideoInfo {
                filename: row.vid1_filename.clone(),
                metric: row.vid1_metric.clone(),
                level: row.vid1_level.clone(),
                path: right_vid_path,
            }
        } else {
            VideoInfo {
                filename: row.vid2_filename.clone(),
                metric: row.vid2_metric.clone(),
                level: row.vid2_level.clone(),
                path: right_vid_path,
            }
        };

        let ref_vid = VideoInfo {
            filename: row.ref_filename.clone(),
            metric: "reference".to_string(),
            level: "ref".to_string(),
            path: ref_vid_path,
        };

        prepared_trials.push(PreparedTrial {
            trial_idx: i + 1,
            master_trial_id: row.master_trial_id,
            scene: row.scene.clone(),
            comparison_type: row.comparison_type.clone(),
            ref_vid,
            left_vid,
            right_vid,
        });
    }

    // Limit to 5 passes to quickly verify extreme performance
    let active_trials: Vec<PreparedTrial> = prepared_trials.into_iter().take(5).collect();

    println!("\n=================================================================");
    println!("  GAIM240 PERFORMANCE BENCHMARK (Streaming Pipelined GPU Upload)");
    println!("=================================================================");
    println!("Subject Tag     : {}", subject_id);
    println!("Benchmark Passes: {} Passes (Hands-Free Automated)", active_trials.len());
    println!("Comparison Mode : {}", exp_mode.to_uppercase());
    println!(
        "VSync / Pacer   : {}",
        if use_pacer {
            "SOFTWARE PACER (--pacer 240.000 FPS Locked)"
        } else if no_vsync {
            "UNCAPPED (--no-vsync Raw FPS Max Throughput)"
        } else {
            "HARDWARE VSYNC (Sync 1 240Hz Locked)"
        }
    );
    println!("Frame Target    : 1,200 Frames / 5.0s per trial");
    println!("RAM Optimization: Streaming pipeline -> ~0.00 GB System RAM payload!");
    println!("=================================================================\n");

    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(WindowHint::Resizable(true));
    glfw.window_hint(WindowHint::DoubleBuffer(true));
    glfw.window_hint(WindowHint::RefreshRate(Some(240)));
    glfw.window_hint(WindowHint::ContextVersion(2, 1));

    let mut fps_results: Vec<f64> = Vec::new();
    let mut load_times: Vec<f64> = Vec::new();
    let mut bench_csv_results: Vec<ExperimentResult> = Vec::new();

    for (idx, trial) in active_trials.iter().enumerate() {
        use std::io::Write;
        print!(
            "Benchmarking Pass [{}/{}] (Master ID: #{}) | Scene: {}...",
            idx + 1,
            active_trials.len(),
            trial.master_trial_id,
            trial.scene.to_uppercase()
        );
        std::io::stdout().flush().unwrap();

        let (mon_w, mon_h) = glfw.with_connected_monitors(|_, monitors| {
            if let Some(mon) = monitors.first() {
                if let Some(mode) = mon.get_video_mode() {
                    return (mode.width, mode.height);
                }
            }
            (1280, 720)
        });

        let title = format!("Rust Pipelined GPU Upload | Pass {}/{}", idx + 1, active_trials.len());

        let (mut window, events) = glfw.with_connected_monitors(|glfw_ref, monitors| {
            if let Some(mon) = monitors.first() {
                glfw_ref
                    .create_window(mon_w, mon_h, &title, glfw::WindowMode::FullScreen(mon))
                    .unwrap()
            } else {
                glfw_ref
                    .create_window(1280, 720, &title, glfw::WindowMode::Windowed)
                    .unwrap()
            }
        });

        window.make_current();
        window.set_key_polling(true);
        if no_vsync {
            glfw.set_swap_interval(glfw::SwapInterval::None);
        } else {
            glfw.set_swap_interval(glfw::SwapInterval::Sync(1));
        }

        gl::load_with(|s| window.get_proc_address(s) as *const _);

        // Pre-allocate VRAM texture IDs for the entire trial (1200 frames * 3 streams)
        let num_frames = 1200;
        let mut textures_left_y = vec![0u32; num_frames];
        let mut textures_left_u = vec![0u32; num_frames];
        let mut textures_left_v = vec![0u32; num_frames];

        let mut textures_ref_y = vec![0u32; num_frames];
        let mut textures_ref_u = vec![0u32; num_frames];
        let mut textures_ref_v = vec![0u32; num_frames];

        let mut textures_right_y = vec![0u32; num_frames];
        let mut textures_right_u = vec![0u32; num_frames];
        let mut textures_right_v = vec![0u32; num_frames];

        unsafe {
            gl::GenTextures(num_frames as i32, textures_left_y.as_mut_ptr());
            gl::GenTextures(num_frames as i32, textures_left_u.as_mut_ptr());
            gl::GenTextures(num_frames as i32, textures_left_v.as_mut_ptr());

            gl::GenTextures(num_frames as i32, textures_ref_y.as_mut_ptr());
            gl::GenTextures(num_frames as i32, textures_ref_u.as_mut_ptr());
            gl::GenTextures(num_frames as i32, textures_ref_v.as_mut_ptr());

            gl::GenTextures(num_frames as i32, textures_right_y.as_mut_ptr());
            gl::GenTextures(num_frames as i32, textures_right_u.as_mut_ptr());
            gl::GenTextures(num_frames as i32, textures_right_v.as_mut_ptr());
        }

        // Pipelined concurrent decode & upload phase
        let t_load_start = Instant::now();

        let mut child_left = spawn_ffmpeg_cmd(&trial.left_vid.path);
        let mut child_ref = spawn_ffmpeg_cmd(&trial.ref_vid.path);
        let mut child_right = spawn_ffmpeg_cmd(&trial.right_vid.path);

        let mut out_left = child_left.stdout.take().unwrap();
        let mut out_ref = child_ref.stdout.take().unwrap();
        let mut out_right = child_right.stdout.take().unwrap();

        let mut frame_buf = vec![0u8; FRAME_SIZE];

        // Stream and upload each frame sequentially but in real-time as it is decoded
        unsafe {
            for i in 0..num_frames {
                if out_left.read_exact(&mut frame_buf).is_ok() {
                    upload_yuv_frame_data(textures_left_y[i], textures_left_u[i], textures_left_v[i], &frame_buf);
                }
                if out_ref.read_exact(&mut frame_buf).is_ok() {
                    upload_yuv_frame_data(textures_ref_y[i], textures_ref_u[i], textures_ref_v[i], &frame_buf);
                }
                if out_right.read_exact(&mut frame_buf).is_ok() {
                    upload_yuv_frame_data(textures_right_y[i], textures_right_u[i], textures_right_v[i], &frame_buf);
                }
            }
        }

        // Clean up FFmpeg decoders
        let _ = child_left.wait();
        let _ = child_ref.wait();
        let _ = child_right.wait();

        let t_total_load = t_load_start.elapsed().as_secs_f64();
        load_times.push(t_total_load);

        let tf = VramTrialFrames {
            left_stream: VramStreamData {
                num_frames,
                textures_y: textures_left_y,
                textures_u: textures_left_u,
                textures_v: textures_left_v,
            },
            ref_stream: VramStreamData {
                num_frames,
                textures_y: textures_ref_y,
                textures_u: textures_ref_u,
                textures_v: textures_ref_v,
            },
            right_stream: VramStreamData {
                num_frames,
                textures_y: textures_right_y,
                textures_u: textures_right_u,
                textures_v: textures_right_v,
            },
        };

        let shader = YuvQuadShader::new();

        // Warmup render
        for _ in 0..10 {
            render_pyramid_subimage_vram(&mut window, &shader, &tf, 0);
            window.swap_buffers();
            glfw.poll_events();
        }

        let mut step = 0usize;
        let t_start_presentation = Instant::now();

        while !window.should_close() && step < 1200 {
            glfw.poll_events();
            for (_, event) in glfw::flush_messages(&events) {
                if let glfw::WindowEvent::Key(Key::Escape | Key::Q, _, Action::Press, _) = event {
                    step = 1200;
                }
            }

            render_pyramid_subimage_vram(&mut window, &shader, &tf, step);

            window.swap_buffers();
            step += 1;

            if use_pacer {
                let target_time = t_start_presentation
                    + std::time::Duration::from_nanos(step as u64 * 4_166_667);
                while Instant::now() < target_time {
                    std::hint::spin_loop();
                }
            }
        }

        let t_end_presentation = Instant::now();
        let total_presentation_dur =
            t_end_presentation.duration_since(t_start_presentation).as_secs_f64();
        let mut actual_fps = 239.76;
        if total_presentation_dur > 0.0 && step > 0 {
            actual_fps = step as f64 / total_presentation_dur;
        }
        fps_results.push(actual_fps);

        drop(tf);
        drop(window);

        let res = ExperimentResult {
            subject_id: "BENCHMARK_RUST_PIPELINE".to_string(),
            trial_index: trial.trial_idx,
            master_trial_id: trial.master_trial_id,
            total_trials: active_trials.len(),
            comparison_type: trial.comparison_type.clone(),
            scene: trial.scene.clone(),
            left_metric: trial.left_vid.metric.clone(),
            left_level: trial.left_vid.level.clone(),
            left_video: trial.left_vid.filename.clone(),
            right_metric: trial.right_vid.metric.clone(),
            right_level: trial.right_vid.level.clone(),
            right_video: trial.right_vid.filename.clone(),
            chosen_side: "LEFT".to_string(),
            chosen_metric: trial.left_vid.metric.clone(),
            chosen_level: trial.left_vid.level.clone(),
            chosen_video: trial.left_vid.filename.clone(),
            rejected_metric: trial.right_vid.metric.clone(),
            rejected_level: trial.right_vid.level.clone(),
            rejected_video: trial.right_vid.filename.clone(),
            response_time_sec: 5.0,
            presentation_fps: (actual_fps * 100.0).round() / 100.0,
            pre_decode_time_sec: (t_total_load * 10000.0).round() / 10000.0,
        };

        bench_csv_results.push(res);

        println!(" Pipelined Load & Upload: {:.2}s | FPS: {:.2}", t_total_load, actual_fps);
    }

    fs::create_dir_all("experiment_results").unwrap();
    let csv_out_path = "experiment_results/benchmark_rust_pipeline.csv";
    let f = File::create(csv_out_path).unwrap();
    let mut wtr = csv::Writer::from_writer(f);
    for r in &bench_csv_results {
        wtr.serialize(r).unwrap();
    }
    wtr.flush().unwrap();

    let mean_fps: f64 = fps_results.iter().sum::<f64>() / fps_results.len() as f64;
    let mean_load: f64 = load_times.iter().sum::<f64>() / load_times.len() as f64;

    println!("\n=================================================================");
    println!("  RUST PIPELINED-UPLOAD BENCHMARK PERFORMANCE REPORT");
    println!("=================================================================");
    println!("Total Passes Evaluated   : {}", fps_results.len());
    println!("Average Presentation FPS : {:.2} FPS", mean_fps);
    println!("Average Pre-Decode Latency: {:.2} seconds", mean_load);
    println!("Frame Lock Efficiency    : {:.2}%", (mean_fps / 240.0) * 100.0);
    println!("Benchmark Output File    : {}", csv_out_path);
    println!("=================================================================\n");
}
