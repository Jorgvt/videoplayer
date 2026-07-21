fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let local_lib = format!("{}/lib/usr/lib/x86_64-linux-gnu", manifest_dir);
    println!("cargo:rustc-link-search=native={}", local_lib);
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", local_lib);
}
