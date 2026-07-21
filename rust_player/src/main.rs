use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::CString;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use glfw::{Action, Context, Key, WindowHint};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone)]
struct FrameData {
    width: i32,
    height: i32,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct TrialFrames {
    left_frames: Arc<Vec<FrameData>>,
    ref_frames: Arc<Vec<FrameData>>,
    right_frames: Arc<Vec<FrameData>>,
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
}

fn decode_video_cmd(path: String) -> Arc<Vec<FrameData>> {
    use std::process::Command;
    let ffmpeg_bin = if Path::new("rust_player/lib/usr/bin/ffmpeg").exists() {
        "rust_player/lib/usr/bin/ffmpeg"
    } else if Path::new("lib/usr/bin/ffmpeg").exists() {
        "lib/usr/bin/ffmpeg"
    } else {
        "ffmpeg"
    };
    
    let output = Command::new(ffmpeg_bin)
        .env("LD_LIBRARY_PATH", "rust_player/lib/usr/lib/x86_64-linux-gnu:lib/usr/lib/x86_64-linux-gnu")
        .args(["-hwaccel", "cuda", "-i", &path, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
        .output();
        
    let mut frames_vec = Vec::new();
    if let Ok(out) = output {
        let raw = out.stdout;
        let frame_size = 1280 * 720 * 3;
        let num_frames = raw.len() / frame_size;
        let max_frames = num_frames; // Decode ALL 1,200 frames (full 5.0s video length)
        
        for i in 0..max_frames {
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
    Arc::new(frames_vec)
}

fn decode_trial_parallel(trial: &PreparedTrial) -> TrialFrames {
    let p_left = trial.left_vid.path.clone();
    let p_ref = trial.ref_vid.path.clone();
    let p_right = trial.right_vid.path.clone();
    
    let h_left = thread::spawn(move || decode_video_cmd(p_left));
    let h_ref = thread::spawn(move || decode_video_cmd(p_ref));
    let h_right = thread::spawn(move || decode_video_cmd(p_right));
    
    TrialFrames {
        left_frames: h_left.join().unwrap(),
        ref_frames: h_ref.join().unwrap(),
        right_frames: h_right.join().unwrap(),
    }
}

struct QuadShader {
    program: u32,
    vbo: u32,
}

impl QuadShader {
    fn new() -> Self {
        let vert_code = CString::new("
            #version 120
            attribute vec2 position;
            attribute vec2 texcoord;
            varying vec2 v_texcoord;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_texcoord = texcoord;
            }
        ").unwrap();
        
        let frag_code = CString::new("
            #version 120
            uniform sampler2D u_texture;
            varying vec2 v_texcoord;
            void main() {
                gl_FragColor = texture2D(u_texture, v_texcoord);
            }
        ").unwrap();
        
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
                gl::TEXTURE_2D, 0, gl::RGB as i32, 1280, 720, 0,
                gl::BGR, gl::UNSIGNED_BYTE, std::ptr::null()
            );
        }
    }
    (textures[0], textures[1], textures[2])
}

fn render_pyramid_subimage(
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
        
        // High-Performance TexSubImage2D DMA Upload
        gl::BindTexture(gl::TEXTURE_2D, tex_a);
        gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, fa.width, fa.height, gl::BGR, gl::UNSIGNED_BYTE, fa.data.as_ptr() as *const _);
        
        gl::BindTexture(gl::TEXTURE_2D, tex_ref);
        gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, fref.width, fref.height, gl::BGR, gl::UNSIGNED_BYTE, fref.data.as_ptr() as *const _);
        
        gl::BindTexture(gl::TEXTURE_2D, tex_c);
        gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, fc.width, fc.height, gl::BGR, gl::UNSIGNED_BYTE, fc.data.as_ptr() as *const _);
        
        shader.draw_quad(-0.5, 0.0, 0.5, 1.0, tex_ref);
        shader.draw_quad(-1.0, -1.0, 0.0, 0.0, tex_a);
        shader.draw_quad(0.0, -1.0, 1.0, 0.0, tex_c);
    }
}

fn hash_subject(subject_id: &str) -> u64 {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    subject_id.hash(&mut s);
    s.finish()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut subject_id = "P01".to_string();
    let mut exp_mode = "full".to_string();
    let mut quick_mode = false;
    let mut borderless = false;
    let mut no_vsync = false;
    let mut use_pacer = false;
    
    for arg in &args[1..] {
        if arg.starts_with("--subject=") {
            subject_id = arg.trim_start_matches("--subject=").to_string();
        } else if arg.starts_with("--mode=") {
            exp_mode = arg.trim_start_matches("--mode=").to_lowercase();
        } else if arg == "--quick" {
            quick_mode = true;
        } else if arg == "--no-vsync" || arg == "--novsync" || arg == "--uncapped" {
            no_vsync = true;
        } else if arg == "--pacer" || arg == "--pace-240" {
            no_vsync = true;
            use_pacer = true;
        } else if arg == "--borderless" {
            borderless = true;
        } else if !arg.starts_with("--") {
            subject_id = arg.clone();
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
    
    let filtered_trials: Vec<MasterTrial> = master_trials.into_iter().filter(|t| {
        if exp_mode == "intra" {
            t.comparison_type == "INTRA-DISTORTION"
        } else if exp_mode == "inter" {
            t.comparison_type == "INTER-DISTORTION"
        } else {
            true
        }
    }).collect();
    
    let seed = hash_subject(&subject_id);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut shuffled_master = filtered_trials;
    shuffled_master.shuffle(&mut rng);
    
    let mut prepared_trials: Vec<PreparedTrial> = Vec::new();
    for (i, row) in shuffled_master.iter().enumerate() {
        let flip: bool = rand::Rng::gen(&mut rng);
        
        let left_vid = if flip {
            VideoInfo { filename: row.vid2_filename.clone(), metric: row.vid2_metric.clone(), level: row.vid2_level.clone(), path: row.vid2_path.clone() }
        } else {
            VideoInfo { filename: row.vid1_filename.clone(), metric: row.vid1_metric.clone(), level: row.vid1_level.clone(), path: row.vid1_path.clone() }
        };
        
        let right_vid = if flip {
            VideoInfo { filename: row.vid1_filename.clone(), metric: row.vid1_metric.clone(), level: row.vid1_level.clone(), path: row.vid1_path.clone() }
        } else {
            VideoInfo { filename: row.vid2_filename.clone(), metric: row.vid2_metric.clone(), level: row.vid2_level.clone(), path: row.vid2_path.clone() }
        };
        
        let ref_vid = VideoInfo { filename: row.ref_filename.clone(), metric: "reference".to_string(), level: "ref".to_string(), path: row.ref_path.clone() };
        
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
    
    let total_trials_count = prepared_trials.len();
    
    fs::create_dir_all("experiment_results").unwrap();
    let res_csv_path = format!("experiment_results/experiment_{}.csv", subject_id);
    let mut completed_ids: HashSet<usize> = HashSet::new();
    let mut existing_results: Vec<ExperimentResult> = Vec::new();
    
    if Path::new(&res_csv_path).exists() {
        if let Ok(mut rdr) = csv::Reader::from_path(&res_csv_path) {
            for res in rdr.deserialize::<ExperimentResult>().flatten() {
                completed_ids.insert(res.master_trial_id);
                existing_results.push(res);
            }
        }
    }
    
    let uncompleted: Vec<PreparedTrial> = prepared_trials.into_iter()
        .filter(|t| !completed_ids.contains(&t.master_trial_id))
        .collect();
        
    let active_trials: Vec<PreparedTrial> = if quick_mode {
        uncompleted.into_iter().take(10).collect()
    } else {
        uncompleted
    };
        
    println!("\n=================================================================");
    println!("  GAIM240 HUMAN VISUAL PERCEPTION EXPERIMENT SUITE (Rust 240Hz)");
    println!("=================================================================");
    println!("Participant ID  : {}", subject_id);
    println!("Total Active    : {}", active_trials.len());
    println!("Experiment Mode : {}", exp_mode.to_uppercase());
    println!("Playback Mode   : {}", if quick_mode { "QUICK (10 trials limit)" } else { "FULL (All trials)" });
    println!("VSync / Pacer   : {}", 
        if use_pacer { "SOFTWARE PACER (--pacer 240.000 FPS Locked)" }
        else if no_vsync { "UNCAPPED (--no-vsync Raw FPS Max Throughput)" }
        else { "HARDWARE VSYNC (Sync 1 240Hz Locked)" }
    );
    println!("Architecture    : Pre-Decoded RAM/VRAM Pipelining");
    println!("=================================================================\n");
    
    if active_trials.is_empty() {
        println!("All trials for subject '{}' are already completed!", subject_id);
        return;
    }
    
    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(WindowHint::Resizable(true));
    glfw.window_hint(WindowHint::DoubleBuffer(true));
    glfw.window_hint(WindowHint::RefreshRate(Some(240)));
    glfw.window_hint(WindowHint::ContextVersion(2, 1));
    
    let mut final_results = existing_results;
    
    for (idx, trial) in active_trials.iter().enumerate() {
        use std::io::Write;
        print!("Executing Trial [{}/{}] (Master ID: #{}) | Scene: {}...", 
            idx + 1, active_trials.len(), trial.master_trial_id, trial.scene.to_uppercase());
        std::io::stdout().flush().unwrap();
        
        let t0 = Instant::now();
        
        // 1. Pre-decode trial frames across CPU cores while screen is black / loading (~2 seconds)
        let tf = decode_trial_parallel(trial);
        let t_load = t0.elapsed().as_secs_f64();
        
        println!(" Pre-Decoded ({:.2}s) -> Presenting Locked 239.76 FPS Pyramid", t_load);
        
        let (mon_w, mon_h) = glfw.with_connected_monitors(|_, monitors| {
            if let Some(mon) = monitors.first() {
                if let Some(mode) = mon.get_video_mode() {
                    return (mode.width, mode.height);
                }
            }
            (1280, 720)
        });
        
        let title = format!("Rust Perception Experiment | Subject: {} | Trial {}/{}", subject_id, idx + 1, active_trials.len());
        
        let (mut window, events) = if borderless {
            glfw.window_hint(WindowHint::Decorated(false));
            let (mut w, e) = glfw.create_window(mon_w, mon_h, &title, glfw::WindowMode::Windowed).unwrap();
            w.set_pos(0, 0);
            (w, e)
        } else {
            glfw.with_connected_monitors(|glfw_ref, monitors| {
                if let Some(mon) = monitors.first() {
                    glfw_ref.create_window(mon_w, mon_h, &title, glfw::WindowMode::FullScreen(mon)).unwrap()
                } else {
                    glfw_ref.create_window(1280, 720, &title, glfw::WindowMode::Windowed).unwrap()
                }
            })
        };
        
        window.make_current();
        window.set_key_polling(true);
        if no_vsync {
            glfw.set_swap_interval(glfw::SwapInterval::None);
        } else {
            glfw.set_swap_interval(glfw::SwapInterval::Sync(1));
        }
        
        gl::load_with(|s| window.get_proc_address(s) as *const _);
        
        let shader = QuadShader::new();
        let (tex_a, tex_ref, tex_c) = create_preallocated_textures();
        
        if tf.left_frames.is_empty() || tf.ref_frames.is_empty() || tf.right_frames.is_empty() {
            println!(" Error: Could not decode video frames for trial #{}. Skipping.", trial.master_trial_id);
            continue;
        }
        
        let fa_0 = &tf.left_frames[0];
        let fref_0 = &tf.ref_frames[0];
        let fc_0 = &tf.right_frames[0];
        
        // 60-frame static warm-up (X11 unredirection & 240Hz sync)
        for _ in 0..60 {
            render_pyramid_subimage(&mut window, &shader, tex_a, tex_ref, tex_c, fa_0, fref_0, fc_0);
            window.swap_buffers();
            glfw.poll_events();
        }
        
        let start_time = Instant::now();
        let mut swap_timestamps: Vec<Instant> = Vec::new();
        let mut step = 0usize;
        let mut choice: Option<String> = None;
        let mut quit = false;
        
        // 2. Presentation Loop
        while !window.should_close() && choice.is_none() && !quit {
            glfw.poll_events();
            for (_, event) in glfw::flush_messages(&events) {
                if let glfw::WindowEvent::Key(Key::Left | Key::A | Key::Kp1, _, Action::Press, _) = event {
                    choice = Some("LEFT".to_string());
                } else if let glfw::WindowEvent::Key(Key::Right | Key::D | Key::Kp2, _, Action::Press, _) = event {
                    choice = Some("RIGHT".to_string());
                } else if let glfw::WindowEvent::Key(Key::Escape | Key::Q, _, Action::Press, _) = event {
                    quit = true;
                }
            }
            
            let fa = &tf.left_frames[step % tf.left_frames.len()];
            let fref = &tf.ref_frames[step % tf.ref_frames.len()];
            let fc = &tf.right_frames[step % tf.right_frames.len()];
            
            render_pyramid_subimage(&mut window, &shader, tex_a, tex_ref, tex_c, fa, fref, fc);
            
            swap_timestamps.push(Instant::now());
            window.swap_buffers();
            step += 1;
            
            // Global drift-compensating nanosecond frame pacer for exact 240.000 FPS (--pacer mode)
            if use_pacer {
                let target_time = start_time + std::time::Duration::from_nanos(step as u64 * 4_166_667);
                while Instant::now() < target_time {
                    std::hint::spin_loop();
                }
            }
        }
        
        let response_time = start_time.elapsed().as_secs_f64();
        
        let mut actual_fps = 239.76;
        if swap_timestamps.len() > 1 {
            let total_dur = swap_timestamps.last().unwrap().duration_since(*swap_timestamps.first().unwrap()).as_secs_f64();
            if total_dur > 0.0 {
                actual_fps = (swap_timestamps.len() - 1) as f64 / total_dur;
            }
        }
        
        unsafe {
            let textures = [tex_a, tex_ref, tex_c];
            gl::DeleteTextures(3, textures.as_ptr());
        }
        drop(tf); // Release CPU RAM immediately after trial ends
        drop(window);
        
        if quit || choice.is_none() {
            println!("\nExperiment stopped early by user. Progress saved.");
            break;
        }
        
        let chosen_side = choice.unwrap();
        let (chosen_vid, rejected_vid) = if chosen_side == "LEFT" {
            (&trial.left_vid, &trial.right_vid)
        } else {
            (&trial.right_vid, &trial.left_vid)
        };
        
        let res = ExperimentResult {
            subject_id: subject_id.clone(),
            trial_index: trial.trial_idx,
            master_trial_id: trial.master_trial_id,
            total_trials: total_trials_count,
            comparison_type: trial.comparison_type.clone(),
            scene: trial.scene.clone(),
            left_metric: trial.left_vid.metric.clone(),
            left_level: trial.left_vid.level.clone(),
            left_video: trial.left_vid.filename.clone(),
            right_metric: trial.right_vid.metric.clone(),
            right_level: trial.right_vid.level.clone(),
            right_video: trial.right_vid.filename.clone(),
            chosen_side,
            chosen_metric: chosen_vid.metric.clone(),
            chosen_level: chosen_vid.level.clone(),
            chosen_video: chosen_vid.filename.clone(),
            rejected_metric: rejected_vid.metric.clone(),
            rejected_level: rejected_vid.level.clone(),
            rejected_video: rejected_vid.filename.clone(),
            response_time_sec: (response_time * 10000.0).round() / 10000.0,
            presentation_fps: (actual_fps * 100.0).round() / 100.0,
        };
        
        println!(" Chose {} ({}) in {:.2}s (FPS: {:.2})", res.chosen_side, res.chosen_level, res.response_time_sec, res.presentation_fps);
        
        final_results.push(res);
        
        let f = File::create(&res_csv_path).unwrap();
        let mut wtr = csv::Writer::from_writer(f);
        for r in &final_results {
            wtr.serialize(r).unwrap();
        }
        wtr.flush().unwrap();
    }
    
    println!("\n=================================================================");
    println!("  Rust Perception Experiment Session Complete!");
    println!("=================================================================");
}
