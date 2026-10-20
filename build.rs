//! Build script for `algebra_linear`.
//!
//! The default build needs nothing from here: the crate is pure Rust, compiles no
//! C and links no library, which is what lets it build for `wasm32-unknown-unknown`.
//!
//! The `dev_lapack` feature is a development-only harness: it compiles
//! `c_code/reference_lapack.c` against a locally built LAPACK 3.12.1 and links it,
//! so the parity tests can compare the kernels against the reference. It is not
//! for release — set `ALGEBRA_LINEAR_DEV_LAPACK_ROOT` to a CMake build directory
//! that holds `include/lapack.h` and `lib/liblapack.so`.
//!
//! This script fails loudly when that directory is absent or incomplete, and pins
//! the build it linked with an `rpath`. It prints nothing on success: which build a
//! test binary runs against is answered by `ldd` on that binary, and a build log is
//! not the place for good news.

fn main() {
    #[cfg(feature = "dev_lapack")]
    dev_lapack::compile();

    #[cfg(not(feature = "dev_lapack"))]
    {
        println!("cargo:rerun-if-changed=build.rs");
    }
}

/// The reference LAPACK, for the parity harness only.
#[cfg(feature = "dev_lapack")]
mod dev_lapack {
    use std::path::Path;

    /// Compiles the reference bindings against a locally built LAPACK and links it.
    ///
    /// # Layout expected
    ///
    /// A CMake build directory of LAPACK 3.12.1 with `include/lapack.h` and
    /// `lib/liblapack.so`. Point `ALGEBRA_LINEAR_DEV_LAPACK_ROOT` at it. There is
    /// deliberately no default: recording one would put a developer's directory
    /// layout into released source.
    pub fn compile() {
        let root = std::env::var("ALGEBRA_LINEAR_DEV_LAPACK_ROOT").unwrap_or_else(|_| {
            panic!(
                "algebra_linear: the `dev_lapack` feature needs ALGEBRA_LINEAR_DEV_LAPACK_ROOT set \
                 to the CMake build directory of a LAPACK 3.12.1 build (the one containing include/ \
                 and lib/)"
            )
        });
        let include = Path::new(&root).join("include");
        let lib = Path::new(&root).join("lib");

        if !include.join("lapack.h").is_file() {
            panic!(
                "algebra_linear: the `dev_lapack` feature needs {}/lapack.h; set \
                 ALGEBRA_LINEAR_DEV_LAPACK_ROOT to the CMake build directory of a LAPACK 3.12.1 build",
                include.display()
            );
        }
        if !lib.join("liblapack.so").is_file() {
            panic!(
                "algebra_linear: the `dev_lapack` feature needs {}/liblapack.so; set \
                 ALGEBRA_LINEAR_DEV_LAPACK_ROOT to the CMake build directory of a LAPACK 3.12.1 build",
                lib.display()
            );
        }

        cc::Build::new()
            .file("c_code/reference_lapack.c")
            .include("c_code")
            .include(&include)
            // The reference is built without warnings-as-errors assumptions of its
            // own; keep its diagnostics out of the crate's build log.
            .warnings(false)
            .compile("reference_lapack");

        println!("cargo:rustc-link-search=native={}", lib.display());
        // LAPACK needs BLAS; CBLAS is not used here (nothing calls `cblas_*`).
        for lib_name in ["lapack", "blas"] {
            println!("cargo:rustc-link-lib=dylib={lib_name}");
        }
        // The reference is Fortran, so its runtime is needed.
        println!("cargo:rustc-link-lib=dylib=gfortran");
        // Make the test binary load the LAPACK build this script was pointed at,
        // rather than whichever `liblapack.so` the loader finds first.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());

        println!("cargo:rerun-if-env-changed=ALGEBRA_LINEAR_DEV_LAPACK_ROOT");
        // The header holds the prototypes; `cc` does not track includes on its own.
        println!("cargo:rerun-if-changed=c_code/reference.h");
        println!("cargo:rerun-if-changed=c_code/reference_lapack.c");
    }
}
