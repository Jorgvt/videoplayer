fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    if cfg!(target_os = "linux") {
        // --- Linux: link against bundled .so files extracted from .deb packages ---
        let local_lib = format!("{}/lib/usr/lib/x86_64-linux-gnu", manifest_dir);
        println!("cargo:rustc-link-search=native={}", local_lib);
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
        // Embed the bundled lib dir into the binary's rpath so it is found at runtime
        // without needing LD_LIBRARY_PATH to be set manually.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", local_lib);
    } else if cfg!(target_os = "windows") {
        // --- Windows: link against DLLs placed in lib/windows/bin ---
        // Place mpv-2.dll, avformat-60.dll, avcodec-60.dll, etc. in that directory.
        // Download from: https://sourceforge.net/projects/mpv-player-windows/
        //                https://www.gyan.dev/ffmpeg/builds/
        let win_dll_dir = format!("{}/lib/windows/bin", manifest_dir);
        println!("cargo:rustc-link-search=native={}", win_dll_dir);
    }
}

