use std::collections::HashSet;
use std::env;
use std::ffi::CString;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use std::io::Read;

use glfw::{Action, Context, Key, WindowHint};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const Y_SIZE: usize = WIDTH * HEIGHT;
const UV_WIDTH: usize = WIDTH;
const UV_HEIGHT: usize = HEIGHT;
const UV_SIZE: usize = Y_SIZE;
const FRAME_SIZE: usize = Y_SIZE + UV_SIZE + UV_SIZE; // YUV444P frame size = 2,764,800 bytes
const PRELOAD_LIMIT: usize = 1200; // Shock absorber buffer size to maintain 240Hz under hardware limits

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
struct VideoStreamData {
    num_frames: usize,
    data: Arc<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct TrialFrames {
    left_stream: Arc<VideoStreamData>,
    ref_stream: Arc<VideoStreamData>,
    right_stream: Arc<VideoStreamData>,
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

fn spawn_ffmpeg_cmd(path: &str) -> std::process::Child {
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

    let mut cmd = std::process::Command::new(ffmpeg_bin);

    #[cfg(target_os = "linux")]
    cmd.env(
        "LD_LIBRARY_PATH",
        "rust_player/lib/usr/lib/x86_64-linux-gnu:lib/usr/lib/x86_64-linux-gnu",
    );

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x00004000;
        cmd.creation_flags(BELOW_NORMAL_PRIORITY_CLASS);

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
            "-stream_loop",
            "-1",
            "-i",
            path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv444p",
            "pipe:1",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("Failed to spawn ffmpeg")
}

fn decode_video_cmd(path: String) -> Arc<VideoStreamData> {
    use std::process::Command;

    // Locate the ffmpeg binary: prefer bundled copies, fall back to system PATH.
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

    // Build the subprocess with the correct library search path for each platform.
    let mut cmd = Command::new(ffmpeg_bin);

    #[cfg(target_os = "linux")]
    cmd.env(
        "LD_LIBRARY_PATH",
        "rust_player/lib/usr/lib/x86_64-linux-gnu:lib/usr/lib/x86_64-linux-gnu",
    );

    #[cfg(target_os = "windows")]
    {
        // On Windows, DLLs are found via PATH. Prepend the bundled DLL directories.
        let dll_dir1 = "rust_player/lib/windows/bin";
        let dll_dir2 = "lib/windows/bin";
        let current_path = std::env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{};{};{}", dll_dir1, dll_dir2, current_path));
    }

    let mut raw_data = Vec::with_capacity(1200 * FRAME_SIZE);

    #[cfg(target_os = "windows")]
    {
        use std::net::TcpListener;
        use std::process::Stdio;
        use std::os::windows::process::CommandExt;

        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x00004000;
        cmd.creation_flags(BELOW_NORMAL_PRIORITY_CLASS);

        let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind TCP listener");
        let port = listener.local_addr().unwrap().port();

        cmd.args([
            "-hwaccel",
            "cuda",
            "-c:v",
            "hevc_cuvid",
            "-i",
            &path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv444p",
            &format!("tcp://127.0.0.1:{}", port),
        ]);

        if let Ok(mut child) = cmd.stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
            if let Ok((mut stream, _)) = listener.accept() {
                std::io::copy(&mut stream, &mut raw_data).ok();
            }
            child.wait().ok();
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        cmd.args([
            "-hwaccel",
            "cuda",
            "-c:v",
            "hevc_cuvid",
            "-i",
            &path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv444p",
            "pipe:1",
        ]);

        if let Ok(mut child) = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            if let Some(mut stdout) = child.stdout.take() {
                std::io::copy(&mut stdout, &mut raw_data).ok();
            }
            child.wait().ok();
        }
    }

    let num_frames = raw_data.len() / FRAME_SIZE;

    Arc::new(VideoStreamData {
        num_frames,
        data: Arc::new(raw_data),
    })
}

fn decode_trial_parallel(trial: &PreparedTrial) -> TrialFrames {
    let p_left = trial.left_vid.path.clone();
    let p_ref = trial.ref_vid.path.clone();
    let p_right = trial.right_vid.path.clone();

    let h_left = thread::spawn(move || decode_video_cmd(p_left));
    let h_ref = thread::spawn(move || decode_video_cmd(p_ref));
    let h_right = thread::spawn(move || decode_video_cmd(p_right));

    TrialFrames {
        left_stream: h_left.join().unwrap(),
        ref_stream: h_ref.join().unwrap(),
        right_stream: h_right.join().unwrap(),
    }
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

struct YuvTextures {
    y: u32,
    u: u32,
    v: u32,
}

fn create_yuv_textures() -> YuvTextures {
    let mut textures = [0u32; 3];
    unsafe {
        gl::Enable(gl::TEXTURE_2D);
        gl::GenTextures(3, textures.as_mut_ptr());

        // Y Texture (1280x720)
        gl::BindTexture(gl::TEXTURE_2D, textures[0]);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RED as i32, WIDTH as i32, HEIGHT as i32, 0,
            gl::RED, gl::UNSIGNED_BYTE, std::ptr::null()
        );

        // U Texture (640x360)
        gl::BindTexture(gl::TEXTURE_2D, textures[1]);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RED as i32, UV_WIDTH as i32, UV_HEIGHT as i32, 0,
            gl::RED, gl::UNSIGNED_BYTE, std::ptr::null()
        );

        // V Texture (640x360)
        gl::BindTexture(gl::TEXTURE_2D, textures[2]);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RED as i32, UV_WIDTH as i32, UV_HEIGHT as i32, 0,
            gl::RED, gl::UNSIGNED_BYTE, std::ptr::null()
        );
    }
    YuvTextures {
        y: textures[0],
        u: textures[1],
        v: textures[2],
    }
}

fn upload_yuv_frame(tex: &YuvTextures, raw_data: &[u8], frame_idx: usize) {
    let frame_offset = frame_idx * FRAME_SIZE;
    if frame_offset + FRAME_SIZE > raw_data.len() {
        return;
    }

    let y_ptr = &raw_data[frame_offset];
    let u_ptr = &raw_data[frame_offset + Y_SIZE];
    let v_ptr = &raw_data[frame_offset + Y_SIZE + UV_SIZE];

    unsafe {
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, tex.y);
        gl::TexSubImage2D(
            gl::TEXTURE_2D, 0, 0, 0, WIDTH as i32, HEIGHT as i32,
            gl::RED, gl::UNSIGNED_BYTE, y_ptr as *const _ as *const _,
        );

        gl::ActiveTexture(gl::TEXTURE1);
        gl::BindTexture(gl::TEXTURE_2D, tex.u);
        gl::TexSubImage2D(
            gl::TEXTURE_2D, 0, 0, 0, UV_WIDTH as i32, UV_HEIGHT as i32,
            gl::RED, gl::UNSIGNED_BYTE, u_ptr as *const _ as *const _,
        );

        gl::ActiveTexture(gl::TEXTURE2);
        gl::BindTexture(gl::TEXTURE_2D, tex.v);
        gl::TexSubImage2D(
            gl::TEXTURE_2D, 0, 0, 0, UV_WIDTH as i32, UV_HEIGHT as i32,
            gl::RED, gl::UNSIGNED_BYTE, v_ptr as *const _ as *const _,
        );
    }
}

fn upload_yuv_single_frame(tex: &YuvTextures, frame: &[u8]) {
    if frame.len() < FRAME_SIZE {
        return;
    }
    let y_ptr = &frame[0];
    let u_ptr = &frame[Y_SIZE];
    let v_ptr = &frame[Y_SIZE + UV_SIZE];

    unsafe {
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, tex.y);
        gl::TexSubImage2D(
            gl::TEXTURE_2D, 0, 0, 0, WIDTH as i32, HEIGHT as i32,
            gl::RED, gl::UNSIGNED_BYTE, y_ptr as *const _ as *const _,
        );

        gl::ActiveTexture(gl::TEXTURE1);
        gl::BindTexture(gl::TEXTURE_2D, tex.u);
        gl::TexSubImage2D(
            gl::TEXTURE_2D, 0, 0, 0, UV_WIDTH as i32, UV_HEIGHT as i32,
            gl::RED, gl::UNSIGNED_BYTE, u_ptr as *const _ as *const _,
        );

        gl::ActiveTexture(gl::TEXTURE2);
        gl::BindTexture(gl::TEXTURE_2D, tex.v);
        gl::TexSubImage2D(
            gl::TEXTURE_2D, 0, 0, 0, UV_WIDTH as i32, UV_HEIGHT as i32,
            gl::RED, gl::UNSIGNED_BYTE, v_ptr as *const _ as *const _,
        );
    }
}

fn render_pyramid_on_the_fly(
    window: &mut glfw::Window,
    shader: &YuvQuadShader,
    tex_a: &YuvTextures,
    tex_ref: &YuvTextures,
    tex_c: &YuvTextures,
    frame_left: &[u8],
    frame_ref: &[u8],
    frame_right: &[u8],
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

    upload_yuv_single_frame(tex_a, frame_left);
    upload_yuv_single_frame(tex_ref, frame_ref);
    upload_yuv_single_frame(tex_c, frame_right);

    unsafe {
        gl::Viewport(0, 0, w, h);
        gl::ClearColor(0.02, 0.02, 0.03, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);

        gl::Viewport(x_offset, y_offset, w_view, h_view);

        shader.draw_quad(-0.5, 0.0, 0.5, 1.0, tex_ref.y, tex_ref.u, tex_ref.v);
        shader.draw_quad(-1.0, -1.0, 0.0, 0.0, tex_a.y, tex_a.u, tex_a.v);
        shader.draw_quad(0.0, -1.0, 1.0, 0.0, tex_c.y, tex_c.u, tex_c.v);
    }
}

fn render_pyramid_subimage(
    window: &mut glfw::Window,
    shader: &YuvQuadShader,
    tex_a: &YuvTextures,
    tex_ref: &YuvTextures,
    tex_c: &YuvTextures,
    stream_a: &VideoStreamData,
    stream_ref: &VideoStreamData,
    stream_c: &VideoStreamData,
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

    let idx_a = if stream_a.num_frames > 0 { frame_idx % stream_a.num_frames } else { 0 };
    let idx_ref = if stream_ref.num_frames > 0 { frame_idx % stream_ref.num_frames } else { 0 };
    let idx_c = if stream_c.num_frames > 0 { frame_idx % stream_c.num_frames } else { 0 };

    upload_yuv_frame(tex_a, &stream_a.data, idx_a);
    upload_yuv_frame(tex_ref, &stream_ref.data, idx_ref);
    upload_yuv_frame(tex_c, &stream_c.data, idx_c);

    unsafe {
        gl::Viewport(0, 0, w, h);
        gl::ClearColor(0.02, 0.02, 0.03, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);

        gl::Viewport(x_offset, y_offset, w_view, h_view);

        shader.draw_quad(-0.5, 0.0, 0.5, 1.0, tex_ref.y, tex_ref.u, tex_ref.v);
        shader.draw_quad(-1.0, -1.0, 0.0, 0.0, tex_a.y, tex_a.u, tex_a.v);
        shader.draw_quad(0.0, -1.0, 1.0, 0.0, tex_c.y, tex_c.u, tex_c.v);
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
    let mut subject_id = "ANONYMOUS_SUBJECT".to_string();
    let mut exp_mode = "intra".to_string();
    let mut quick_mode = false;
    let mut no_vsync = false;
    let mut use_pacer = false;
    let mut borderless = false;

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

    let total_trials_count = prepared_trials.len();
    fs::create_dir_all("experiment_results").unwrap();
    let res_csv_path = format!("experiment_results/{}_results.csv", subject_id);

    let mut existing_results: Vec<ExperimentResult> = Vec::new();
    let mut completed_ids = HashSet::new();

    if Path::new(&res_csv_path).exists() {
        if let Ok(mut rdr) = csv::Reader::from_path(&res_csv_path) {
            for result in rdr.deserialize::<ExperimentResult>().filter_map(|r| r.ok()) {
                completed_ids.insert(result.master_trial_id);
                existing_results.push(result);
            }
        }
    }

    let uncompleted: Vec<PreparedTrial> = prepared_trials
        .into_iter()
        .filter(|t| !completed_ids.contains(&t.master_trial_id))
        .collect();

    let active_trials: Vec<PreparedTrial> = if quick_mode {
        uncompleted.into_iter().take(10).collect()
    } else {
        uncompleted
    };

    println!("\n=================================================================");
    println!("  GAIM240 HUMAN VISUAL PERCEPTION EXPERIMENT SUITE (Rust YUV444p)");
    println!("=================================================================");
    println!("Participant ID  : {}", subject_id);
    println!("Total Active    : {}", active_trials.len());
    println!("Experiment Mode : {}", exp_mode.to_uppercase());
    println!("Playback Mode   : {}", if quick_mode { "QUICK (10 trials limit)" } else { "FULL (All trials)" });
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
    println!("Architecture    : YUV444p Planar (On-the-Fly GPU hardware Decoding)");
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

    // Create window once before the loop
    let (mon_w, mon_h) = glfw.with_connected_monitors(|_, monitors| {
        if let Some(mon) = monitors.first() {
            if let Some(mode) = mon.get_video_mode() {
                return (mode.width, mode.height);
            }
        }
        (1280, 720)
    });

    let (mut window, events) = if borderless {
        glfw.window_hint(WindowHint::Decorated(false));
        let (mut w, e) = glfw
            .create_window(mon_w, mon_h, "GAIM240 Player", glfw::WindowMode::Windowed)
            .unwrap();
        w.set_pos(0, 0);
        (w, e)
    } else {
        glfw.with_connected_monitors(|glfw_ref, monitors| {
            if let Some(mon) = monitors.first() {
                glfw_ref
                    .create_window(mon_w, mon_h, "GAIM240 Player", glfw::WindowMode::FullScreen(mon))
                    .unwrap()
            } else {
                glfw_ref
                    .create_window(1280, 720, "GAIM240 Player", glfw::WindowMode::Windowed)
                    .unwrap()
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

    let shader = YuvQuadShader::new();
    let tex_a = create_yuv_textures();
    let tex_ref = create_yuv_textures();
    let tex_c = create_yuv_textures();

    // Session-wide GPU Warmup Phase (60 dummy black frames render)
    {
        use std::io::Write;
        print!("Warming up GPU and compiling shaders... ");
        std::io::stdout().flush().ok();
        let dummy_data = vec![0u8; FRAME_SIZE];
        let y_ptr = &dummy_data[0];
        let u_ptr = &dummy_data[Y_SIZE];
        let v_ptr = &dummy_data[Y_SIZE + UV_SIZE];
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, tex_a.y);
            gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, WIDTH as i32, HEIGHT as i32, gl::RED, gl::UNSIGNED_BYTE, y_ptr as *const _ as *const _);
            gl::ActiveTexture(gl::TEXTURE1);
            gl::BindTexture(gl::TEXTURE_2D, tex_a.u);
            gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, UV_WIDTH as i32, UV_HEIGHT as i32, gl::RED, gl::UNSIGNED_BYTE, u_ptr as *const _ as *const _);
            gl::ActiveTexture(gl::TEXTURE2);
            gl::BindTexture(gl::TEXTURE_2D, tex_a.v);
            gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, UV_WIDTH as i32, UV_HEIGHT as i32, gl::RED, gl::UNSIGNED_BYTE, v_ptr as *const _ as *const _);
        }
        for _ in 0..60 {
            unsafe {
                gl::Viewport(0, 0, mon_w as i32, mon_h as i32);
                gl::ClearColor(0.02, 0.02, 0.03, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT);
                shader.draw_quad(-0.5, 0.0, 0.5, 1.0, tex_a.y, tex_a.u, tex_a.v);
                shader.draw_quad(-1.0, -1.0, 0.0, 0.0, tex_a.y, tex_a.u, tex_a.v);
                shader.draw_quad(0.0, -1.0, 1.0, 0.0, tex_a.y, tex_a.u, tex_a.v);
            }
            window.swap_buffers();
            glfw.poll_events();
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
        println!("Done.");
    }

    // Synchronously pre-decode Trial 1 in parallel threads to eliminate any background activity during Pass 1
    println!("Pre-decoding Trial 1 (fully) at session startup... ");
    let t_pre_start = Instant::now();
    let first_trial = active_trials[0].clone();
    let tf_first = decode_trial_parallel(&first_trial);
    let t_pre_done = t_pre_start.elapsed().as_secs_f64();
    println!("Done ({:.2}s)", t_pre_done);

    let mut final_results = existing_results;

    for (idx, trial) in active_trials.iter().enumerate() {
        use std::io::Write;
        print!(
            "Executing Trial [{}/{}] (Master ID: #{}) | Scene: {}...",
            idx + 1,
            active_trials.len(),
            trial.master_trial_id,
            trial.scene.to_uppercase()
        );
        std::io::stdout().flush().unwrap();

        // Update window title dynamically
        let title = format!(
            "Rust Perception Experiment | Subject: {} | Trial {}/{}",
            subject_id,
            idx + 1,
            active_trials.len()
        );
        window.set_title(&title);

        let t_load_start = Instant::now();

        let mut child_left: Option<std::process::Child> = None;
        let mut child_ref: Option<std::process::Child> = None;
        let mut child_right: Option<std::process::Child> = None;

        let mut rx_left = None;
        let mut rx_ref = None;
        let mut rx_right = None;

        let mut recycle_left = None;
        let mut recycle_ref = None;
        let mut recycle_right = None;

        if idx > 0 {
            child_left = Some(spawn_ffmpeg_cmd(&trial.left_vid.path));
            child_ref = Some(spawn_ffmpeg_cmd(&trial.ref_vid.path));
            child_right = Some(spawn_ffmpeg_cmd(&trial.right_vid.path));

            let decoded_left = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let decoded_ref = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let decoded_right = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

            macro_rules! make_stream {
                ($child:expr, $counter:expr) => {{
                    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(PRELOAD_LIMIT + 3);
                    let (recycle_tx, recycle_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(PRELOAD_LIMIT + 3);
                    for _ in 0..(PRELOAD_LIMIT + 3) {
                        recycle_tx.send(vec![0u8; FRAME_SIZE]).ok();
                    }
                    let mut out = $child.as_mut().unwrap().stdout.take().unwrap();
                    let counter_clone = std::sync::Arc::clone(&$counter);
                    std::thread::spawn(move || {
                        while let Ok(mut buf) = recycle_rx.recv() {
                            if out.read_exact(&mut buf).is_err() {
                                break;
                            }
                            if ready_tx.send(buf).is_err() {
                                break;
                            }
                            counter_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        }
                    });
                    (ready_rx, recycle_tx)
                }};
            }

            let (rl, recl) = make_stream!(child_left, decoded_left);
            let (rr, recr) = make_stream!(child_ref, decoded_ref);
            let (rrr, recrr) = make_stream!(child_right, decoded_right);

            rx_left = Some(rl);
            rx_ref = Some(rr);
            rx_right = Some(rrr);

            recycle_left = Some(recl);
            recycle_ref = Some(recr);
            recycle_right = Some(recrr);

            while decoded_left.load(std::sync::atomic::Ordering::SeqCst) < PRELOAD_LIMIT
                || decoded_ref.load(std::sync::atomic::Ordering::SeqCst) < PRELOAD_LIMIT
                || decoded_right.load(std::sync::atomic::Ordering::SeqCst) < PRELOAD_LIMIT
            {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        let t_load = if idx == 0 {
            t_pre_done
        } else {
            t_load_start.elapsed().as_secs_f64()
        };

        if idx > 0 && (child_left.is_none() || child_ref.is_none() || child_right.is_none()) {
            println!(
                " Error: Could not decode video frames for trial #{}. Skipping.",
                trial.master_trial_id
            );
            continue;
        }

        if idx == 0 {
            println!(" Pre-Decoded ({:.2}s)", t_load);
        } else {
            println!(" On-the-Fly GPU Decoded ({:.2}s)", t_load);
        }

        let start_time = Instant::now();
        let mut swap_timestamps: Vec<Instant> = Vec::new();
        let mut step = 0usize;
        let mut choice: Option<String> = None;
        let mut quit = false;

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

            let mut buf_left = None;
            let mut buf_ref = None;
            let mut buf_right = None;

            let frame_left: &[u8] = if idx == 0 {
                let frame_idx = step % tf_first.left_stream.num_frames;
                let offset = frame_idx * FRAME_SIZE;
                &tf_first.left_stream.data[offset .. offset + FRAME_SIZE]
            } else {
                buf_left = Some(match rx_left.as_ref().unwrap().recv() { Ok(b) => b, Err(_) => break });
                buf_left.as_ref().unwrap()
            };

            let frame_ref: &[u8] = if idx == 0 {
                let frame_idx = step % tf_first.ref_stream.num_frames;
                let offset = frame_idx * FRAME_SIZE;
                &tf_first.ref_stream.data[offset .. offset + FRAME_SIZE]
            } else {
                buf_ref = Some(match rx_ref.as_ref().unwrap().recv() { Ok(b) => b, Err(_) => break });
                buf_ref.as_ref().unwrap()
            };

            let frame_right: &[u8] = if idx == 0 {
                let frame_idx = step % tf_first.right_stream.num_frames;
                let offset = frame_idx * FRAME_SIZE;
                &tf_first.right_stream.data[offset .. offset + FRAME_SIZE]
            } else {
                buf_right = Some(match rx_right.as_ref().unwrap().recv() { Ok(b) => b, Err(_) => break });
                buf_right.as_ref().unwrap()
            };

            render_pyramid_on_the_fly(
                &mut window,
                &shader,
                &tex_a,
                &tex_ref,
                &tex_c,
                frame_left,
                frame_ref,
                frame_right,
            );

            if idx > 0 {
                recycle_left.as_ref().unwrap().send(buf_left.unwrap()).ok();
                recycle_ref.as_ref().unwrap().send(buf_ref.unwrap()).ok();
                recycle_right.as_ref().unwrap().send(buf_right.unwrap()).ok();
            }

            swap_timestamps.push(Instant::now());
            window.swap_buffers();
            step += 1;

            if use_pacer {
                let target_time = start_time + std::time::Duration::from_nanos(step as u64 * 4_166_667);
                while Instant::now() < target_time {
                    std::hint::spin_loop();
                }
            }
        }

        let response_time = start_time.elapsed().as_secs_f64();

        let mut actual_fps = 239.76;
        let mut dropped_frames = 0;
        if swap_timestamps.len() > 1 {
            let total_dur = swap_timestamps
                .last()
                .unwrap()
                .duration_since(*swap_timestamps.first().unwrap())
                .as_secs_f64();
            if total_dur > 0.0 {
                actual_fps = (swap_timestamps.len() - 1) as f64 / total_dur;
            }
            for i in 1..swap_timestamps.len() {
                let diff = swap_timestamps[i].duration_since(swap_timestamps[i - 1]).as_secs_f64() * 1000.0;
                if diff > 6.25 {
                    dropped_frames += 1;
                }
            }
        }

        if idx > 0 {
            let _ = child_left.as_mut().unwrap().kill();
            let _ = child_ref.as_mut().unwrap().kill();
            let _ = child_right.as_mut().unwrap().kill();
        }

        // Textures and window persist across trials

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

        println!(
            " Chose {} ({}) in {:.2}s (FPS: {:.2} | Drops: {})",
            res.chosen_side, res.chosen_level, res.response_time_sec, res.presentation_fps, dropped_frames
        );

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
