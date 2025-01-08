use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use core::panic;
use flate2::read::GzDecoder;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;
use tar::Archive;

#[derive(Parser)]
#[command(author, version, about = "Process LC0 training data from tar files")]
struct Args {
    /// Path to the tar file containing .gz training data
    #[arg(short, long)]
    tar_path: String,

    #[arg(short, long, default_value_t = 10)]
    num_entries: usize,
}

#[derive(Debug)]
struct TrainingData {
    version: u32,
    _input_format: u32,
    _probabilities: Vec<f32>,
    planes: Vec<u64>,
    castling_us_ooo: u8,
    castling_us_oo: u8,
    castling_them_ooo: u8,
    castling_them_oo: u8,
    _side_to_move_or_enpassant: u8,
    rule50_count: u8,
    _invariance_info: u8,
    _dummy: u8,
    _root_q: f32,
    best_q: f32,
    _root_d: f32,
    best_d: f32,
    _root_m: f32,
    _best_m: f32,
    _plies_left: f32,
    _result_q: f32,
    _result_d: f32,
    _played_q: f32,
    _played_d: f32,
    _played_m: f32,
    _orig_q: f32,
    _orig_d: f32,
    _orig_m: f32,
    _visits: u32,
    _played_idx: u16,
    best_idx: u16,
    _policy_kld: f32,
    _reserved: u32,
}

// For some reason, lc0 reverses the bits in the bytes of the bitboard before
// storing them in the training data.
// https://github.com/search?q=repo%3ALeelaChessZero%2Flc0+ReverseBitsInBytes&type=code
fn reverse_bits_in_bytes(x: u64) -> u64 {
    let mut v = x;
    v = ((v >> 1) & 0x5555555555555555) | ((v & 0x5555555555555555) << 1);
    v = ((v >> 2) & 0x3333333333333333) | ((v & 0x3333333333333333) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0F) | ((v & 0x0F0F0F0F0F0F0F0F) << 4);
    v
}

impl TrainingData {
    fn read_from<R: Read>(mut reader: R) -> io::Result<Self> {
        let version = reader.read_u32::<LittleEndian>()?;
        let input_format = reader.read_u32::<LittleEndian>()?;

        let mut probabilities = vec![0.0; 1858];
        for prob in probabilities.iter_mut() {
            *prob = reader.read_f32::<LittleEndian>()?;
        }

        let mut planes = vec![0; 104];
        for plane in planes.iter_mut() {
            *plane = reverse_bits_in_bytes(reader.read_u64::<LittleEndian>()?);
        }

        Ok(TrainingData {
            version,
            _input_format: input_format,
            _probabilities: probabilities,
            planes,
            castling_us_ooo: reader.read_u8()?,
            castling_us_oo: reader.read_u8()?,
            castling_them_ooo: reader.read_u8()?,
            castling_them_oo: reader.read_u8()?,
            _side_to_move_or_enpassant: reader.read_u8()?,
            rule50_count: reader.read_u8()?,
            _invariance_info: reader.read_u8()?,
            _dummy: reader.read_u8()?,
            _root_q: reader.read_f32::<LittleEndian>()?,
            best_q: reader.read_f32::<LittleEndian>()?,
            _root_d: reader.read_f32::<LittleEndian>()?,
            best_d: reader.read_f32::<LittleEndian>()?,
            _root_m: reader.read_f32::<LittleEndian>()?,
            _best_m: reader.read_f32::<LittleEndian>()?,
            _plies_left: reader.read_f32::<LittleEndian>()?,
            _result_q: reader.read_f32::<LittleEndian>()?,
            _result_d: reader.read_f32::<LittleEndian>()?,
            _played_q: reader.read_f32::<LittleEndian>()?,
            _played_d: reader.read_f32::<LittleEndian>()?,
            _played_m: reader.read_f32::<LittleEndian>()?,
            _orig_q: reader.read_f32::<LittleEndian>()?,
            _orig_d: reader.read_f32::<LittleEndian>()?,
            _orig_m: reader.read_f32::<LittleEndian>()?,
            _visits: reader.read_u32::<LittleEndian>()?,
            _played_idx: reader.read_u16::<LittleEndian>()?,
            best_idx: reader.read_u16::<LittleEndian>()?,
            _policy_kld: reader.read_f32::<LittleEndian>()?,
            _reserved: reader.read_u32::<LittleEndian>()?,
        })
    }
}

fn process_position(data: TrainingData) {
    assert_eq!(data.version, 6);
    assert!(data._side_to_move_or_enpassant < 2);
}

fn process_gz_file<R: Read>(reader: R) -> io::Result<()> {
    let mut gz = GzDecoder::new(reader);

    while let Ok(data) = TrainingData::read_from(&mut gz) {
        process_position(data);
    }

    Ok(())
}

fn process_tar_file<P: AsRef<Path>>(path: P) -> io::Result<()> {
    let file = File::open(path)?;
    let mut archive = Archive::new(file);

    for entry in archive.entries()? {
        let entry = entry?;
        if !entry.path()?.to_string_lossy().ends_with(".gz") {
            continue;
        }

        process_gz_file(entry)?;
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    process_tar_file(&args.tar_path)
}
