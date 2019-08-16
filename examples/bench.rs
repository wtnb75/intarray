extern crate clap;
extern crate digest;

// use blake2::Blake2b;
use clap::{App, Arg};
use digest::Digest;
use hex;
use intarray::IntArray;
use sha2::Sha256;
use std::mem;

fn hash<D: Digest>(input: &str, output: &mut [u8]) {
    let mut hasher = D::new();
    hasher.input(input.as_bytes());
    output.copy_from_slice(hasher.result().as_slice());
}

fn main() {
    let app = App::new("bench")
        .version("0.0.1")
        .arg(
            Arg::with_name("bits")
                .short("b")
                .long("bits")
                .takes_value(true)
                .default_value("4"),
        )
        .arg(
            Arg::with_name("length")
                .short("l")
                .long("length")
                .takes_value(true)
                .default_value("1024"),
        )
        .arg(Arg::with_name("count"));
    let matches = app.get_matches();
    let mut bits: usize = 0;
    if let Some(o) = matches.value_of("bits") {
        bits = o.parse::<usize>().unwrap();
        println!("{} bits", bits);
        assert!(bits != 0, "0 bits");
    }
    let mut length: usize = 0;
    if let Some(o) = matches.value_of("length") {
        length = o.parse::<usize>().unwrap();
        println!("length={} ", length);
    }
    let mut v = IntArray::new(bits, length);
    v.fill_random();
    println!("v={}", v.datasize());

    let mut result: [u8; 32] = [0; 32];
    hash::<Sha256>("hello world", &mut result);
    println!("result={}", hex::encode(result));
    unsafe {
        for i in 0..(32 - 8) {
            let v: u64 = mem::transmute_copy(&result[i]);
            println!("h[{}]={:x}", i, v);
        }
    }
}
