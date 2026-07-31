use std::env;
use std::ffi::CString;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use glfw::{Action, Context, Key, WindowHint};
use serde::{Deserialize, Serialize};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const Y_SIZE: usize = WIDTH * HEIGHT;
const UV_WIDTH: usize = WIDTH / 2;
const UV_HEIGHT: usize = HEIGHT / 2;
const UV_SIZE: usize = UV_WIDTH * UV_HEIGHT;
const FRAME_SIZE: usize = Y_SIZE + UV_SIZE + UV_SIZE;

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
struct SingleTrialResult {
    #[serde(rename = "LeftPath")]
    left_path: String,
    #[serde(rename = "RightPath")]
    right_path: String,
    #[serde(rename = "ChosenSide")]
    chosen_side: String,
    #[serde(rename = "ResponseTime_sec")]
    response_time_sec: f64,
    #[serde(rename = "PresentationFPS")]
    presentation_fps: f64,
}

fn decode_video_cmd(path: String) -> Arc<VideoStreamData> {
    use std::process::Command;
    let ffmpeg_bin = if Path::new("rust_player/lib/usr/bin/ffmpeg").exists() {
        "rust_player/lib/usr/bin/ffmpeg"
    } else if Path::new("lib/usr/bin/ffmpeg").exists() {
        "lib/usr/bin/ffmpeg"
    } else {
        "ffmpeg"
    };

    let output = Command::new(ffmpeg_bin)
        .env(
            "LD_LIBRARY_PATH",
            "rust_player/lib/usr/lib/x86_64-linux-gnu:lib/usr/lib/x86_64-linux-gnu",
        )
        .args([
            "-hwaccel",
            "cuda",
            "-i",
            &path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ])
        .output();

    let mut raw_data = Vec::new();
    if let Ok(out) = output {
        raw_data = out.stdout;
    }

    let num_frames = raw_data.len() / FRAME_SIZE;

    Arc::new(VideoStreamData {
        num_frames,
        data: Arc::new(raw_data),
    })
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

        gl::BindTexture(gl::TEXTURE_2D, textures[0]);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RED as i32, WIDTH as i32, HEIGHT as i32, 0,
            gl::RED, gl::UNSIGNED_BYTE, std::ptr::null()
        );

        gl::BindTexture(gl::TEXTURE_2D, textures[1]);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RED as i32, UV_WIDTH as i32, UV_HEIGHT as i32, 0,
            gl::RED, gl::UNSIGNED_BYTE, std::ptr::null()
        );

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

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut left_path = String::new();
    let mut ref_path = String::new();
    let mut right_path = String::new();
    let mut out_csv_path = "asap_trial_result.csv".to_string();
    let mut no_vsync = false;
    let mut use_pacer = true;
    let mut borderless = false;

    for arg in &args[1..] {
        if arg.starts_with("--left=") {
            left_path = arg.trim_start_matches("--left=").to_string();
        } else if arg.starts_with("--ref=") {
            ref_path = arg.trim_start_matches("--ref=").to_string();
        } else if arg.starts_with("--right=") {
            right_path = arg.trim_start_matches("--right=").to_string();
        } else if arg.starts_with("--out=") {
            out_csv_path = arg.trim_start_matches("--out=").to_string();
        } else if arg == "--no-vsync" || arg == "--uncapped" {
            no_vsync = true;
            use_pacer = false;
        } else if arg == "--pacer" || arg == "--pace-240" {
            use_pacer = true;
        } else if arg == "--borderless" {
            borderless = true;
        }
    }

    if left_path.is_empty() || ref_path.is_empty() || right_path.is_empty() {
        println!("Error: Must provide --left=<path> --ref=<path> --right=<path>");
        return;
    }

    let p_left = left_path.clone();
    let p_ref = ref_path.clone();
    let p_right = right_path.clone();

    let h_left = thread::spawn(move || decode_video_cmd(p_left));
    let h_ref = thread::spawn(move || decode_video_cmd(p_ref));
    let h_right = thread::spawn(move || decode_video_cmd(p_right));

    let tf = TrialFrames {
        left_stream: h_left.join().unwrap(),
        ref_stream: h_ref.join().unwrap(),
        right_stream: h_right.join().unwrap(),
    };

    if tf.left_stream.num_frames == 0
        || tf.ref_stream.num_frames == 0
        || tf.right_stream.num_frames == 0
    {
        println!("Error: Failed to decode one or more video streams.");
        return;
    }

    let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
    glfw.window_hint(WindowHint::Resizable(true));
    glfw.window_hint(WindowHint::DoubleBuffer(true));
    glfw.window_hint(WindowHint::RefreshRate(Some(240)));
    glfw.window_hint(WindowHint::ContextVersion(2, 1));

    let (mon_w, mon_h) = glfw.with_connected_monitors(|_, monitors| {
        if let Some(mon) = monitors.first() {
            if let Some(mode) = mon.get_video_mode() {
                return (mode.width, mode.height);
            }
        }
        (1280, 720)
    });

    let title = "GAIM240 240Hz 2AFC | Press [A] Left vs [D] Right | Esc Quit".to_string();

    let (mut window, events) = if borderless {
        glfw.window_hint(WindowHint::Decorated(false));
        let (mut w, e) = glfw
            .create_window(mon_w, mon_h, &title, glfw::WindowMode::Windowed)
            .unwrap();
        w.set_pos(0, 0);
        (w, e)
    } else {
        glfw.with_connected_monitors(|glfw_ref, monitors| {
            if let Some(mon) = monitors.first() {
                glfw_ref
                    .create_window(mon_w, mon_h, &title, glfw::WindowMode::FullScreen(mon))
                    .unwrap()
            } else {
                glfw_ref
                    .create_window(1280, 720, &title, glfw::WindowMode::Windowed)
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

    // Warmup render
    for _ in 0..30 {
        render_pyramid_subimage(
            &mut window,
            &shader,
            &tex_a,
            &tex_ref,
            &tex_c,
            &tf.left_stream,
            &tf.ref_stream,
            &tf.right_stream,
            0,
        );
        window.swap_buffers();
        glfw.poll_events();
    }

    let start_time = Instant::now();
    let mut swap_timestamps: Vec<Instant> = Vec::new();
    let mut step = 0usize;
    let mut choice: Option<String> = None;

    while !window.should_close() && choice.is_none() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            if let glfw::WindowEvent::Key(Key::Left | Key::A | Key::Kp1, _, Action::Press, _) = event {
                choice = Some("LEFT".to_string());
            } else if let glfw::WindowEvent::Key(Key::Right | Key::D | Key::Kp2, _, Action::Press, _) = event {
                choice = Some("RIGHT".to_string());
            } else if let glfw::WindowEvent::Key(Key::Escape | Key::Q, _, Action::Press, _) = event {
                window.set_should_close(true);
            }
        }

        render_pyramid_subimage(
            &mut window,
            &shader,
            &tex_a,
            &tex_ref,
            &tex_c,
            &tf.left_stream,
            &tf.ref_stream,
            &tf.right_stream,
            step,
        );

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
    if swap_timestamps.len() > 1 {
        let total_dur = swap_timestamps
            .last()
            .unwrap()
            .duration_since(*swap_timestamps.first().unwrap())
            .as_secs_f64();
        if total_dur > 0.0 {
            actual_fps = (swap_timestamps.len() - 1) as f64 / total_dur;
        }
    }

    unsafe {
        let textures = [
            tex_a.y, tex_a.u, tex_a.v,
            tex_ref.y, tex_ref.u, tex_ref.v,
            tex_c.y, tex_c.u, tex_c.v,
        ];
        gl::DeleteTextures(9, textures.as_ptr());
    }

    if let Some(ch) = choice {
        let res = SingleTrialResult {
            left_path,
            right_path,
            chosen_side: ch,
            response_time_sec: (response_time * 10000.0).round() / 10000.0,
            presentation_fps: (actual_fps * 100.0).round() / 100.0,
        };

        if let Ok(f) = File::create(&out_csv_path) {
            let mut wtr = csv::Writer::from_writer(f);
            wtr.serialize(&res).unwrap();
            wtr.flush().unwrap();
        }
    }
}
