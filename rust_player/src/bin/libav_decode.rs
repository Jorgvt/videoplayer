use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::time::Instant;

use libloading::Library;

// C Types
type AVFormatContextPtr = *mut c_void;
type AVCodecContextPtr = *mut c_void;
type AVCodecPtr = *const c_void;
type AVPacketPtr = *mut c_void;
type AVFramePtr = *mut c_void;

fn main() {
    let test_file = "/home/jv495/Datasets/GAIM240/zeroday_restir_level2.mp4";
    println!("=== Testing Direct In-Process libavcodec Decoding in Rust ===");
    println!("File: {}\n", test_file);

    unsafe {
        let libavformat = Library::new("/usr/lib/x86_64-linux-gnu/libavformat.so.60")
            .expect("Failed to load libavformat.so.60");
        let libavcodec = Library::new("/usr/lib/x86_64-linux-gnu/libavcodec.so.60")
            .expect("Failed to load libavcodec.so.60");

        // Load symbols
        let avformat_open_input: libloading::Symbol<
            unsafe extern "C" fn(
                *mut AVFormatContextPtr,
                *const c_char,
                *mut c_void,
                *mut *mut c_void,
            ) -> c_int,
        > = libavformat.get(b"avformat_open_input\0").unwrap();

        let avformat_find_stream_info: libloading::Symbol<
            unsafe extern "C" fn(AVFormatContextPtr, *mut *mut c_void) -> c_int,
        > = libavformat.get(b"avformat_find_stream_info\0").unwrap();

        let avformat_close_input: libloading::Symbol<
            unsafe extern "C" fn(*mut AVFormatContextPtr),
        > = libavformat.get(b"avformat_close_input\0").unwrap();

        let av_read_frame: libloading::Symbol<
            unsafe extern "C" fn(AVFormatContextPtr, AVPacketPtr) -> c_int,
        > = libavformat.get(b"av_read_frame\0").unwrap();

        let avcodec_find_decoder: libloading::Symbol<
            unsafe extern "C" fn(c_int) -> AVCodecPtr,
        > = libavcodec.get(b"avcodec_find_decoder\0").unwrap();

        let avcodec_alloc_context3: libloading::Symbol<
            unsafe extern "C" fn(AVCodecPtr) -> AVCodecContextPtr,
        > = libavcodec.get(b"avcodec_alloc_context3\0").unwrap();

        let avcodec_parameters_to_context: libloading::Symbol<
            unsafe extern "C" fn(AVCodecContextPtr, *const c_void) -> c_int,
        > = libavcodec.get(b"avcodec_parameters_to_context\0").unwrap();

        let avcodec_open2: libloading::Symbol<
            unsafe extern "C" fn(AVCodecContextPtr, AVCodecPtr, *mut *mut c_void) -> c_int,
        > = libavcodec.get(b"avcodec_open2\0").unwrap();

        let av_packet_alloc: libloading::Symbol<unsafe extern "C" fn() -> AVPacketPtr> =
            libavcodec.get(b"av_packet_alloc\0").unwrap();

        let av_frame_alloc: libloading::Symbol<unsafe extern "C" fn() -> AVFramePtr> =
            libavcodec.get(b"av_frame_alloc\0").unwrap();

        let avcodec_send_packet: libloading::Symbol<
            unsafe extern "C" fn(AVCodecContextPtr, AVPacketPtr) -> c_int,
        > = libavcodec.get(b"avcodec_send_packet\0").unwrap();

        let avcodec_receive_frame: libloading::Symbol<
            unsafe extern "C" fn(AVCodecContextPtr, AVFramePtr) -> c_int,
        > = libavcodec.get(b"avcodec_receive_frame\0").unwrap();

        let av_packet_free: libloading::Symbol<unsafe extern "C" fn(*mut AVPacketPtr)> =
            libavcodec.get(b"av_packet_free\0").unwrap();

        let av_frame_free: libloading::Symbol<unsafe extern "C" fn(*mut AVFramePtr)> =
            libavcodec.get(b"av_frame_free\0").unwrap();

        let avcodec_free_context: libloading::Symbol<
            unsafe extern "C" fn(*mut AVCodecContextPtr),
        > = libavcodec.get(b"avcodec_free_context\0").unwrap();

        let c_path = CString::new(test_file).unwrap();
        let mut fmt_ctx: AVFormatContextPtr = std::ptr::null_mut();

        let t0 = Instant::now();

        if avformat_open_input(&mut fmt_ctx, c_path.as_ptr(), std::ptr::null_mut(), std::ptr::null_mut()) != 0 {
            println!("Error opening video file.");
            return;
        }

        if avformat_find_stream_info(fmt_ctx, std::ptr::null_mut()) < 0 {
            println!("Error finding stream info.");
            avformat_close_input(&mut fmt_ctx);
            return;
        }

        // Locate video stream (codec_type == 0 for AVMEDIA_TYPE_VIDEO)
        // Offset 48 in AVFormatContext is nb_streams, offset 56 is streams**
        let nb_streams = *(fmt_ctx.add(48) as *const u32);
        let streams_ptr = *(fmt_ctx.add(56) as *const *const *const c_void);

        let mut video_stream_idx = -1i32;
        let mut codec_par_ptr: *const c_void = std::ptr::null();

        for i in 0..nb_streams {
            let stream = *streams_ptr.add(i as usize);
            // offset 8 in AVStream is codecpar pointer
            let par = *(stream.add(8) as *const *const c_void);
            let codec_type = *(par as *const i32); // offset 0 in AVCodecParameters is enum AVMediaType
            if codec_type == 0 {
                video_stream_idx = i as i32;
                codec_par_ptr = par;
                break;
            }
        }

        if video_stream_idx < 0 || codec_par_ptr.is_null() {
            println!("Error: No video stream found.");
            avformat_close_input(&mut fmt_ctx);
            return;
        }

        // offset 4 in AVCodecParameters is enum AVCodecID
        let codec_id = *(codec_par_ptr.add(4) as *const i32);
        let decoder = avcodec_find_decoder(codec_id);
        if decoder.is_null() {
            println!("Error: Decoder not found.");
            avformat_close_input(&mut fmt_ctx);
            return;
        }

        let mut codec_ctx = avcodec_alloc_context3(decoder);
        avcodec_parameters_to_context(codec_ctx, codec_par_ptr);

        // Enable multi-threaded decoding in AVCodecContext
        // offset 408 is thread_count, offset 412 is thread_type (1=frame, 2=slice, 3=both)
        *(codec_ctx.add(408) as *mut i32) = 0; // Auto thread count
        *(codec_ctx.add(412) as *mut i32) = 3; // FRAME + SLICE

        if avcodec_open2(codec_ctx, decoder, std::ptr::null_mut()) < 0 {
            println!("Error opening codec.");
            avcodec_free_context(&mut codec_ctx);
            avformat_close_input(&mut fmt_ctx);
            return;
        }

        let mut pkt = av_packet_alloc();
        let mut frame = av_frame_alloc();
        let mut decoded_frames = 0usize;

        while av_read_frame(fmt_ctx, pkt) == 0 {
            // offset 32 in AVPacket is stream_index
            let stream_idx = *(pkt.add(32) as *const i32);
            if stream_idx == video_stream_idx {
                if avcodec_send_packet(codec_ctx, pkt) == 0 {
                    while avcodec_receive_frame(codec_ctx, frame) == 0 {
                        decoded_frames += 1;
                    }
                }
            }
        }

        // Flush remaining frames
        avcodec_send_packet(codec_ctx, std::ptr::null_mut());
        while avcodec_receive_frame(codec_ctx, frame) == 0 {
            decoded_frames += 1;
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 { decoded_frames as f64 / elapsed } else { 0.0 };

        println!(
            "In-Process libavcodec: Decoded {:4} frames in {:6.3} s | Speed = {:6.2} FPS ({:.2}x)",
            decoded_frames,
            elapsed,
            fps,
            fps / 240.0
        );

        av_frame_free(&mut frame);
        av_packet_free(&mut pkt);
        avcodec_free_context(&mut codec_ctx);
        avformat_close_input(&mut fmt_ctx);
    }
}
