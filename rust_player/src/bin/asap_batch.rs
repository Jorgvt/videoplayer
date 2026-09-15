use std::collections::HashSet;
use std::env;
use std::ffi::CString;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use glfw::{Action, Context, Key, WindowHint};
use serde::{Deserialize, Serialize};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const RGB_FRAME_SIZE: usize = WIDTH * HEIGHT * 3; // 2,764,800 bytes
const Y_SIZE: usize = WIDTH * HEIGHT;
const UV_WIDTH: usize = WIDTH / 2;
const UV_HEIGHT: usize = HEIGHT / 2;
const UV_SIZE: usize = UV_WIDTH * UV_HEIGHT;
const FRAME_SIZE: usize = Y_SIZE + UV_SIZE + UV_SIZE; // 1,382,400 bytes

#[derive(Debug, Clone)]
struct VideoStreamData {
    num_frames: usize,
    data: Arc<Vec<u8>>,
    is_rgb: bool,
}

#[derive(Debug, Clone)]
struct TrialFrames {
    left_stream: Arc<VideoStreamData>,
    ref_stream: Arc<VideoStreamData>,
    right_stream: Arc<VideoStreamData>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct BatchRow {
    #[serde(rename = "TrialNumber")]
    trial_number: usize,
    #[serde(rename = "SubjectID")]
    subject_id: String,
    #[serde(rename = "Scene")]
    scene: String,
    #[serde(rename = "RefFilename")]
    ref_filename: String,
    #[serde(rename = "RefPath")]
    ref_path: String,
    #[serde(rename = "LeftCondition")]
    left_condition: String,
    #[serde(rename = "LeftFilename")]
    left_filename: String,
    #[serde(rename = "LeftMetric")]
    left_metric: String,
    #[serde(rename = "LeftLevel")]
    left_level: String,
    #[serde(rename = "LeftPath")]
    left_path: String,
    #[serde(rename = "RightCondition")]
    right_condition: String,
    #[serde(rename = "RightFilename")]
    right_filename: String,
    #[serde(rename = "RightMetric")]
    right_metric: String,
    #[serde(rename = "RightLevel")]
    right_level: String,
    #[serde(rename = "RightPath")]
    right_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrialResultRecord {
    #[serde(rename = "SubjectID")]
    subject_id: String,
    #[serde(rename = "TrialNumber")]
    trial_number: usize,
    #[serde(rename = "Scene")]
    scene: String,
    #[serde(rename = "LeftCondition")]
    left_condition: String,
    #[serde(rename = "RightCondition")]
    right_condition: String,
    #[serde(rename = "ChosenSide")]
    chosen_side: String,
    #[serde(rename = "ChosenCondition")]
    chosen_condition: String,
    #[serde(rename = "RejectedCondition")]
    rejected_condition: String,
    #[serde(rename = "ResponseTime_sec")]
    response_time_sec: f64,
    #[serde(rename = "PresentationFPS")]
    presentation_fps: f64,
}

fn decode_video_cmd(path: String) -> Arc<VideoStreamData> {
    if path.ends_with(".rgb") || path.ends_with(".raw") {
        let raw_data = std::fs::read(&path).unwrap_or_default();
        let num_frames = raw_data.len() / RGB_FRAME_SIZE;
        return Arc::new(VideoStreamData {
            num_frames,
            data: Arc::new(raw_data),
            is_rgb: true,
        });
    }

    use std::process::Command;

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
        let dll_dir = "rust_player/lib/windows/bin";
        let current_path = std::env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{};{}", dll_dir, current_path));
    }

    let output = cmd
        .args([
            "-hwaccel",
            "cuda",
            "-c:v",
            "hevc_cuvid",
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
        is_rgb: false,
    })
}

fn load_trial_triplet_parallel(left_path: String, ref_path: String, right_path: String) -> TrialFrames {
    let (left_stream, (ref_stream, right_stream)) = rayon::join(
        || decode_video_cmd(left_path),
        || rayon::join(
            || decode_video_cmd(ref_path),
            || decode_video_cmd(right_path)
        )
    );
    TrialFrames {
        left_stream,
        ref_stream,
        right_stream,
    }
}

fn get_font_glyph(c: char) -> [u8; 8] {
    match c {
        '0' => [0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00],
        '1' => [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00],
        '2' => [0x3C, 0x66, 0x06, 0x1C, 0x30, 0x66, 0x7E, 0x00],
        '3' => [0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00],
        '4' => [0x0C, 0x1C, 0x34, 0x64, 0x7E, 0x04, 0x0E, 0x00],
        '5' => [0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00],
        '6' => [0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00],
        '7' => [0x7E, 0x66, 0x0C, 0x18, 0x18, 0x18, 0x18, 0x00],
        '8' => [0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00],
        '9' => [0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00],
        'A' | 'a' => [0x18, 0x3C, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00],
        'B' | 'b' => [0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00],
        'C' | 'c' => [0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00],
        'D' | 'd' => [0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00],
        'E' | 'e' => [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00],
        'F' | 'f' => [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00],
        'G' | 'g' => [0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3A, 0x00],
        'H' | 'h' => [0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00],
        'I' | 'i' => [0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00],
        'J' | 'j' => [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38, 0x00],
        'K' | 'k' => [0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00],
        'L' | 'l' => [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00],
        'M' | 'm' => [0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00],
        'N' | 'n' => [0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00],
        'O' | 'o' => [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00],
        'P' | 'p' => [0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00],
        'Q' | 'q' => [0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00],
        'R' | 'r' => [0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00],
        'S' | 's' => [0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00],
        'T' | 't' => [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00],
        'U' | 'u' => [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00],
        'V' | 'v' => [0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00],
        'W' | 'w' => [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00],
        'X' | 'x' => [0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00],
        'Y' | 'y' => [0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00],
        'Z' | 'z' => [0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00],
        '/' => [0x02, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00],
        '%' => [0x62, 0x64, 0x08, 0x10, 0x20, 0x26, 0x46, 0x00],
        '(' => [0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00],
        ')' => [0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00],
        ':' => [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30],
        _ => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    }
}

fn create_text_texture(text: &str) -> (u32, usize, usize) {
    let scale = 4usize;
    let char_w = 8 * scale;
    let char_h = 8 * scale;
    let width = text.len() * char_w;
    let height = char_h;

    let mut rgba_data = vec![0u8; width * height * 4];

    for (ci, c) in text.chars().enumerate() {
        let glyph = get_font_glyph(c);
        let x_offset = ci * char_w;
        for (row_i, &row_byte) in glyph.iter().enumerate() {
            for col_i in 0..8 {
                let bit_set = ((row_byte >> (7 - col_i)) & 1) == 1;
                if bit_set {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = x_offset + col_i * scale + sx;
                            let py = row_i * scale + sy;
                            let idx = (py * width + px) * 4;
                            if idx + 3 < rgba_data.len() {
                                rgba_data[idx] = 230;
                                rgba_data[idx + 1] = 232;
                                rgba_data[idx + 2] = 238;
                                rgba_data[idx + 3] = 255;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut tex = 0;
    unsafe {
        gl::GenTextures(1, &mut tex);
        gl::BindTexture(gl::TEXTURE_2D, tex);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D,
            0,
            gl::RGBA as i32,
            width as i32,
            height as i32,
            0,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
            rgba_data.as_ptr() as *const _,
        );
    }
    (tex, width, height)
}

struct BorderShader {
    program: u32,
    vbo: u32,
    u_color_loc: i32,
}

impl BorderShader {
    fn new() -> Self {
        let vert_code = CString::new(
            "
            #version 120
            attribute vec2 position;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
        )
        .unwrap();

        let frag_code = CString::new(
            "
            #version 120
            uniform vec4 u_color;
            void main() {
                gl_FragColor = u_color;
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

            let u_color_loc = gl::GetUniformLocation(program, CString::new("u_color").unwrap().as_ptr());

            BorderShader {
                program,
                vbo,
                u_color_loc,
            }
        }
    }

    fn draw_rect_fill(&self, x1: f32, y1: f32, x2: f32, y2: f32) {
        #[repr(C)]
        struct Vertex {
            pos: [f32; 2],
        }

        let vertices: [Vertex; 6] = [
            Vertex { pos: [x1, y1] },
            Vertex { pos: [x2, y1] },
            Vertex { pos: [x2, y2] },
            Vertex { pos: [x1, y1] },
            Vertex { pos: [x2, y2] },
            Vertex { pos: [x1, y2] },
        ];

        unsafe {
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

            gl::DrawArrays(gl::TRIANGLES, 0, 6);
        }
    }

    fn draw_rect_fill_colored(&self, x1: f32, y1: f32, x2: f32, y2: f32, color: [f32; 4]) {
        unsafe {
            gl::UseProgram(self.program);
            gl::Uniform4f(self.u_color_loc, color[0], color[1], color[2], color[3]);
        }
        self.draw_rect_fill(x1, y1, x2, y2);
    }

    fn draw_box_border(
        &self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        w_view: i32,
        h_view: i32,
        thickness_px: f32,
        color: [f32; 4],
    ) {
        let dx = (thickness_px / w_view as f32) * 2.0;
        let dy = (thickness_px / h_view as f32) * 2.0;

        unsafe {
            gl::UseProgram(self.program);
            gl::Uniform4f(self.u_color_loc, color[0], color[1], color[2], color[3]);
        }

        // Top edge
        self.draw_rect_fill(x1, y2 - dy, x2, y2);
        // Bottom edge
        self.draw_rect_fill(x1, y1, x2, y1 + dy);
        // Left edge
        self.draw_rect_fill(x1, y1, x1 + dx, y2);
        // Right edge
        self.draw_rect_fill(x2 - dx, y1, x2, y2);
    }
}

struct RgbQuadShader {
    program: u32,
    vbo: u32,
    u_tex_loc: i32,
}

impl RgbQuadShader {
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
            uniform sampler2D u_tex;
            varying vec2 v_texcoord;
            void main() {
                gl_FragColor = texture2D(u_tex, v_texcoord);
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

            let u_tex_loc = gl::GetUniformLocation(program, CString::new("u_tex").unwrap().as_ptr());

            RgbQuadShader {
                program,
                vbo,
                u_tex_loc,
            }
        }
    }

    fn draw_quad(&self, x1: f32, y1: f32, x2: f32, y2: f32, tex: u32) {
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
            gl::BindTexture(gl::TEXTURE_2D, tex);
            gl::Uniform1i(self.u_tex_loc, 0);

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

fn create_rgb_texture() -> u32 {
    let mut tex = 0;
    unsafe {
        gl::GenTextures(1, &mut tex);
        gl::BindTexture(gl::TEXTURE_2D, tex);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::TexImage2D(
            gl::TEXTURE_2D, 0, gl::RGB as i32, WIDTH as i32, HEIGHT as i32, 0,
            gl::RGB, gl::UNSIGNED_BYTE, std::ptr::null()
        );
    }
    tex
}

fn upload_rgb_frame(tex: u32, raw_data: &[u8], frame_idx: usize) {
    let frame_offset = frame_idx * RGB_FRAME_SIZE;
    if frame_offset + RGB_FRAME_SIZE > raw_data.len() {
        return;
    }
    let ptr = &raw_data[frame_offset];
    unsafe {
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, tex);
        gl::TexSubImage2D(
            gl::TEXTURE_2D, 0, 0, 0, WIDTH as i32, HEIGHT as i32,
            gl::RGB, gl::UNSIGNED_BYTE, ptr as *const _ as *const _,
        );
    }
}

fn render_pyramid_rgb(
    window: &mut glfw::Window,
    shader: &RgbQuadShader,
    border_shader: Option<&BorderShader>,
    tex_a: u32,
    tex_ref: u32,
    tex_c: u32,
    stream_a: &VideoStreamData,
    stream_ref: &VideoStreamData,
    stream_c: &VideoStreamData,
    frame_idx: usize,
    highlight_side: Option<&str>,
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

    upload_rgb_frame(tex_a, &stream_a.data, idx_a);
    upload_rgb_frame(tex_ref, &stream_ref.data, idx_ref);
    upload_rgb_frame(tex_c, &stream_c.data, idx_c);

    unsafe {
        gl::Viewport(0, 0, w, h);
        gl::ClearColor(0.02, 0.02, 0.03, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);

        gl::Viewport(x_offset, y_offset, w_view, h_view);

        shader.draw_quad(-0.5, 0.0, 0.5, 1.0, tex_ref);
        shader.draw_quad(-1.0, -1.0, 0.0, 0.0, tex_a);
        shader.draw_quad(0.0, -1.0, 1.0, 0.0, tex_c);
    }

    if let (Some(side), Some(bs)) = (highlight_side, border_shader) {
        let green_color = [0.0f32, 0.9f32, 0.35f32, 1.0f32];
        let thickness = 8.0f32;
        if side == "LEFT" {
            bs.draw_box_border(-1.0, -1.0, 0.0, 0.0, w_view, h_view, thickness, green_color);
        } else if side == "RIGHT" {
            bs.draw_box_border(0.0, -1.0, 1.0, 0.0, w_view, h_view, thickness, green_color);
        }
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

fn render_pyramid_yuv(
    window: &mut glfw::Window,
    shader: &YuvQuadShader,
    border_shader: Option<&BorderShader>,
    tex_a: &YuvTextures,
    tex_ref: &YuvTextures,
    tex_c: &YuvTextures,
    stream_a: &VideoStreamData,
    stream_ref: &VideoStreamData,
    stream_c: &VideoStreamData,
    frame_idx: usize,
    highlight_side: Option<&str>,
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

    if let (Some(side), Some(bs)) = (highlight_side, border_shader) {
        let green_color = [0.0f32, 0.9f32, 0.35f32, 1.0f32];
        let thickness = 8.0f32;
        if side == "LEFT" {
            bs.draw_box_border(-1.0, -1.0, 0.0, 0.0, w_view, h_view, thickness, green_color);
        } else if side == "RIGHT" {
            bs.draw_box_border(0.0, -1.0, 1.0, 0.0, w_view, h_view, thickness, green_color);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut batch_csv_path = String::new();
    let mut out_csv_path = String::new();
    let mut no_vsync = false;
    let mut use_pacer = true;
    let mut borderless = false;
    let mut feedback_ms = 300u64;
    let mut warmup_ms = 0u64;

    for arg in &args[1..] {
        if arg.starts_with("--batch=") {
            batch_csv_path = arg.trim_start_matches("--batch=").to_string();
        } else if arg.starts_with("--out=") {
            out_csv_path = arg.trim_start_matches("--out=").to_string();
        } else if arg == "--no-vsync" || arg == "--uncapped" {
            no_vsync = true;
            use_pacer = false;
        } else if arg == "--pacer" || arg == "--pace-240" {
            use_pacer = true;
        } else if arg == "--borderless" {
            borderless = true;
        } else if arg.starts_with("--feedback-ms=") {
            if let Ok(val) = arg.trim_start_matches("--feedback-ms=").parse::<u64>() {
                feedback_ms = val;
            }
        } else if arg.starts_with("--warmup-ms=") || arg.starts_with("--min-progress-ms=") {
            let prefix = if arg.starts_with("--warmup-ms=") { "--warmup-ms=" } else { "--min-progress-ms=" };
            if let Ok(val) = arg.trim_start_matches(prefix).parse::<u64>() {
                warmup_ms = val;
            }
        }
    }

    if batch_csv_path.is_empty() {
        println!("Error: Must provide --batch=<path_to_batch.csv>");
        return;
    }

    if out_csv_path.is_empty() {
        out_csv_path = "experiment_results/trials_batch_results.csv".to_string();
    }

    // Read batch trials
    let mut batch_trials: Vec<BatchRow> = Vec::new();
    {
        let mut rdr = csv::Reader::from_path(&batch_csv_path).expect("Failed to open batch CSV");
        for result in rdr.deserialize::<BatchRow>() {
            match result {
                Ok(row) => batch_trials.push(row),
                Err(e) => eprintln!("Warning: Failed to parse batch row: {}", e),
            }
        }
    }

    let total_batch_trials = batch_trials.len();
    if total_batch_trials == 0 {
        println!("Error: No trials found in batch CSV: {}", batch_csv_path);
        return;
    }

    // Ensure parent directory exists for results CSV
    if let Some(parent) = Path::new(&out_csv_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Check for auto-resume from existing results
    let mut completed_numbers = HashSet::new();
    let mut completed_records: Vec<TrialResultRecord> = Vec::new();

    if Path::new(&out_csv_path).exists() {
        if let Ok(mut rdr) = csv::Reader::from_path(&out_csv_path) {
            for result in rdr.deserialize::<TrialResultRecord>().filter_map(|r| r.ok()) {
                completed_numbers.insert(result.trial_number);
                completed_records.push(result);
            }
        }
    }

    let num_completed_initially = completed_numbers.len();
    println!("\n=================================================================");
    println!("  GAIM240 ACTIVE SAMPLING CONTINUOUS BATCH RUNNER (240Hz Rust)");
    println!("=================================================================");
    println!("Batch File           : {}", batch_csv_path);
    println!("Output Results CSV   : {}", out_csv_path);
    println!("Total Batch Trials   : {}", total_batch_trials);
    if num_completed_initially > 0 {
        println!("Previously Completed : {}/{} (Resuming)", num_completed_initially, total_batch_trials);
    }
    println!("Presentation Mode    : {}", if borderless { "BORDERLESS WINDOW" } else { "EXCLUSIVE FULLSCREEN" });
    println!("Pacer / VSync        : {}", if use_pacer { "SOFTWARE PACER (240.0 FPS)" } else if no_vsync { "UNCAPPED MAX FPS" } else { "HARDWARE VSYNC" });
    println!("Visual Feedback      : {} ms green highlight", feedback_ms);
    println!("Progress Card Hold   : {}", if warmup_ms > 0 { format!("{} ms minimum (or until next presentation loads)", warmup_ms) } else { "Dynamic (until next presentation finishes loading in memory)".to_string() });
    println!("=================================================================\n");

    if num_completed_initially >= total_batch_trials {
        println!("Notice: All {} trials in this batch are already completed!", total_batch_trials);
        return;
    }

    // Initialize GLFW and Fullscreen Window ONCE
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
    window.set_cursor_mode(glfw::CursorMode::Hidden);

    if no_vsync {
        glfw.set_swap_interval(glfw::SwapInterval::None);
    } else {
        glfw.set_swap_interval(glfw::SwapInterval::Sync(1));
    }

    gl::load_with(|s| window.get_proc_address(s) as *const _);

    let quad_shader = RgbQuadShader::new();
    let yuv_shader = YuvQuadShader::new();
    let border_shader = BorderShader::new();

    let rgb_tex_a = create_rgb_texture();
    let rgb_tex_ref = create_rgb_texture();
    let rgb_tex_c = create_rgb_texture();

    let yuv_tex_a = create_yuv_textures();
    let yuv_tex_ref = create_yuv_textures();
    let yuv_tex_c = create_yuv_textures();

    let mut user_interrupted = false;

    // Main Single-Window Batch Trial Loop
    for row in &batch_trials {
        let trial_num = row.trial_number;
        if completed_numbers.contains(&trial_num) {
            continue;
        }

        println!("--- Trial [{}/{}] (Scene: {}) ---", trial_num, total_batch_trials, row.scene.to_uppercase());
        println!("  Left  : {}", row.left_condition);
        println!("  Right : {}", row.right_condition);

        // 1. ASSET LOADING & PROGRESS CARD PHASE (Synchronized with next presentation decode)
        // Spawn parallel thread to load the 3 video streams exclusively while progress card is visible
        let (tx, rx) = mpsc::channel::<TrialFrames>();
        let left_p = row.left_path.clone();
        let ref_p = row.ref_path.clone();
        let right_p = row.right_path.clone();

        thread::spawn(move || {
            let tf = load_trial_triplet_parallel(left_p, ref_p, right_p);
            tx.send(tf).ok();
        });

        // Setup Progress Text & Track
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

        let (title_tex, title_w, _) = create_text_texture(&format!("Trial {} / {}", trial_num, total_batch_trials));
        let pct = (trial_num * 100) / total_batch_trials;
        let (pct_tex, pct_w, _) = create_text_texture(&format!("({}%)", pct));

        let hw_title = (0.09 * (title_w as f32 / 32.0) * (9.0 / 16.0)) / 2.0;
        let hw_pct = (0.055 * (pct_w as f32 / 32.0) * (9.0 / 16.0)) / 2.0;
        let fraction = (trial_num as f32 / total_batch_trials as f32).clamp(0.0, 1.0);
        let fill_x = -0.25 + 0.50 * fraction;

        let warmup_start = Instant::now();
        let warmup_dur = std::time::Duration::from_millis(warmup_ms);
        let mut warmup_step = 0usize;
        let mut loaded_frames: Option<TrialFrames> = None;

        // Render progress card dynamically while next presentation is decoding in memory
        while (warmup_step == 0 || warmup_start.elapsed() < warmup_dur || loaded_frames.is_none()) && !window.should_close() {
            glfw.poll_events();
            for (_, event) in glfw::flush_messages(&events) {
                if let glfw::WindowEvent::Key(Key::Escape | Key::Q, _, Action::Press, _) = event {
                    window.set_should_close(true);
                    user_interrupted = true;
                }
            }

            if loaded_frames.is_none() {
                if let Ok(tf) = rx.try_recv() {
                    loaded_frames = Some(tf);
                }
            }

            unsafe {
                gl::Viewport(0, 0, w, h);
                gl::ClearColor(0.02, 0.02, 0.03, 1.0);
                gl::Clear(gl::COLOR_BUFFER_BIT);

                gl::Viewport(x_offset, y_offset, w_view, h_view);

                gl::Enable(gl::BLEND);
                gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

                // 1. Draw Title "Trial X / Y"
                quad_shader.draw_quad(-hw_title, 0.04, hw_title, 0.13, title_tex);

                // 2. Draw Progress Track & Fill
                border_shader.draw_rect_fill_colored(-0.25, -0.025, 0.25, -0.010, [0.12, 0.12, 0.16, 1.0]);
                border_shader.draw_rect_fill_colored(-0.25, -0.025, fill_x, -0.010, [0.0, 0.88, 0.45, 1.0]);

                // 3. Draw Percentage "(Z%)"
                quad_shader.draw_quad(-hw_pct, -0.09, hw_pct, -0.035, pct_tex);

                gl::Disable(gl::BLEND);
            }

            window.swap_buffers();
            warmup_step += 1;

            if use_pacer {
                let target = warmup_start + std::time::Duration::from_nanos(warmup_step as u64 * 4_166_667);
                while Instant::now() < target {
                    std::hint::spin_loop();
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(4));
            }
        }

        unsafe {
            gl::DeleteTextures(1, &title_tex);
            gl::DeleteTextures(1, &pct_tex);
        }

        if user_interrupted || window.should_close() {
            println!("\nSession interrupted by observer before trial {}. Progress saved.", trial_num);
            break;
        }

        let tf = match loaded_frames {
            Some(f) => f,
            None => rx.recv().expect("Failed to receive decoded video frames"),
        };

        if tf.left_stream.num_frames == 0 || tf.ref_stream.num_frames == 0 || tf.right_stream.num_frames == 0 {
            eprintln!("Error: Failed to decode one or more streams for trial #{}. Skipping.", trial_num);
            continue;
        }

        let is_rgb_mode = tf.left_stream.is_rgb && tf.ref_stream.is_rgb && tf.right_stream.is_rgb;

        // 2. ISOLATED 240Hz PLAYBACK PHASE (Zero Background Activity)
        let start_time = Instant::now();
        let mut swap_timestamps: Vec<Instant> = Vec::new();
        let mut step = 0usize;
        let mut choice: Option<String> = None;
        let mut response_time = 0.0f64;

        while !window.should_close() && choice.is_none() {
            glfw.poll_events();
            for (_, event) in glfw::flush_messages(&events) {
                if let glfw::WindowEvent::Key(Key::Left | Key::A | Key::Kp1, _, Action::Press, _) = event {
                    response_time = start_time.elapsed().as_secs_f64();
                    choice = Some("LEFT".to_string());
                } else if let glfw::WindowEvent::Key(Key::Right | Key::D | Key::Kp2, _, Action::Press, _) = event {
                    response_time = start_time.elapsed().as_secs_f64();
                    choice = Some("RIGHT".to_string());
                } else if let glfw::WindowEvent::Key(Key::Escape | Key::Q, _, Action::Press, _) = event {
                    window.set_should_close(true);
                    user_interrupted = true;
                }
            }

            if choice.is_some() || window.should_close() {
                break;
            }

            if is_rgb_mode {
                render_pyramid_rgb(
                    &mut window,
                    &quad_shader,
                    None,
                    rgb_tex_a,
                    rgb_tex_ref,
                    rgb_tex_c,
                    &tf.left_stream,
                    &tf.ref_stream,
                    &tf.right_stream,
                    step,
                    None,
                );
            } else {
                render_pyramid_yuv(
                    &mut window,
                    &yuv_shader,
                    None,
                    &yuv_tex_a,
                    &yuv_tex_ref,
                    &yuv_tex_c,
                    &tf.left_stream,
                    &tf.ref_stream,
                    &tf.right_stream,
                    step,
                    None,
                );
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

        if choice.is_none() && (user_interrupted || window.should_close()) {
            println!("\nSession interrupted during trial {}. Progress saved.", trial_num);
            break;
        }

        // 3. VISUAL FEEDBACK PHASE (300ms Green Selection Border + Frozen Frame)
        if let Some(ref ch) = choice {
            if !window.should_close() && feedback_ms > 0 {
                let freeze_step = step.saturating_sub(1);
                let feedback_start = Instant::now();
                let feedback_dur = std::time::Duration::from_millis(feedback_ms);
                let mut fb_step = step;

                while !window.should_close() && feedback_start.elapsed() < feedback_dur {
                    glfw.poll_events();
                    for (_, event) in glfw::flush_messages(&events) {
                        if let glfw::WindowEvent::Key(Key::Escape | Key::Q, _, Action::Press, _) = event {
                            window.set_should_close(true);
                            user_interrupted = true;
                        }
                    }

                    if is_rgb_mode {
                        render_pyramid_rgb(
                            &mut window,
                            &quad_shader,
                            Some(&border_shader),
                            rgb_tex_a,
                            rgb_tex_ref,
                            rgb_tex_c,
                            &tf.left_stream,
                            &tf.ref_stream,
                            &tf.right_stream,
                            freeze_step,
                            Some(ch.as_str()),
                        );
                    } else {
                        render_pyramid_yuv(
                            &mut window,
                            &yuv_shader,
                            Some(&border_shader),
                            &yuv_tex_a,
                            &yuv_tex_ref,
                            &yuv_tex_c,
                            &tf.left_stream,
                            &tf.ref_stream,
                            &tf.right_stream,
                            freeze_step,
                            Some(ch.as_str()),
                        );
                    }

                    swap_timestamps.push(Instant::now());
                    window.swap_buffers();
                    fb_step += 1;

                    if use_pacer {
                        let target_time = start_time + std::time::Duration::from_nanos(fb_step as u64 * 4_166_667);
                        while Instant::now() < target_time {
                            std::hint::spin_loop();
                        }
                    }
                }
            }

            // Calculate presentation FPS
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

            let chosen_cond = if ch == "LEFT" { row.left_condition.clone() } else { row.right_condition.clone() };
            let rejected_cond = if ch == "LEFT" { row.right_condition.clone() } else { row.left_condition.clone() };

            let record = TrialResultRecord {
                subject_id: row.subject_id.clone(),
                trial_number: trial_num,
                scene: row.scene.clone(),
                left_condition: row.left_condition.clone(),
                right_condition: row.right_condition.clone(),
                chosen_side: ch.clone(),
                chosen_condition: chosen_cond.clone(),
                rejected_condition: rejected_cond.clone(),
                response_time_sec: (response_time * 10000.0).round() / 10000.0,
                presentation_fps: (actual_fps * 100.0).round() / 100.0,
            };

            completed_records.push(record.clone());
            completed_numbers.insert(trial_num);

            println!("  -> Result: Chose {} ({}) in {:.2}s (FPS: {:.1})\n", ch, chosen_cond, response_time, actual_fps);

            // Incrementally append/flush trial to CSV
            let file_exists = Path::new(&out_csv_path).exists() && std::fs::metadata(&out_csv_path).map(|m| m.len() > 0).unwrap_or(false);
            if let Ok(f) = OpenOptions::new().create(true).append(true).open(&out_csv_path) {
                let mut wtr = if file_exists {
                    csv::WriterBuilder::new().has_headers(false).from_writer(f)
                } else {
                    csv::WriterBuilder::new().has_headers(true).from_writer(f)
                };
                wtr.serialize(&record).ok();
                wtr.flush().ok();
            }
        }

        if user_interrupted || window.should_close() {
            break;
        }
    }

    // Cleanup Shaders & Textures
    unsafe {
        gl::DeleteProgram(border_shader.program);
        gl::DeleteBuffers(1, &border_shader.vbo);

        gl::DeleteProgram(quad_shader.program);
        gl::DeleteBuffers(1, &quad_shader.vbo);

        gl::DeleteProgram(yuv_shader.program);
        gl::DeleteBuffers(1, &yuv_shader.vbo);

        let rgb_textures = [rgb_tex_a, rgb_tex_ref, rgb_tex_c];
        gl::DeleteTextures(3, rgb_textures.as_ptr());

        let yuv_textures = [yuv_tex_a.y, yuv_tex_a.u, yuv_tex_a.v, yuv_tex_ref.y, yuv_tex_ref.u, yuv_tex_ref.v, yuv_tex_c.y, yuv_tex_c.u, yuv_tex_c.v];
        gl::DeleteTextures(9, yuv_textures.as_ptr());
    }

    println!("\n=================================================================");
    println!("  BATCH PRESENTATION SESSION FINISHED");
    println!("=================================================================");
    println!("Completed {}/{} trials logged to: {}", completed_records.len(), total_batch_trials, out_csv_path);
    println!("=================================================================\n");
}
