use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use flate2::read::GzDecoder;
use shakmaty::{Bitboard, Board, ByColor, ByRole, Chess, Role, Square};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use tar::Archive;

#[derive(Parser)]
#[command(author, version, about = "Process LC0 training data from tar files")]
struct Args {
    /// Path to the tar file containing .gz training data
    #[arg(short, long)]
    tar_path: String,
}

const MIN_PIECES: u32 = 7;

// This struct loosely follows the format of the training data used by lc0.
//
// https://lczero.org/dev/wiki/training-data-format-versions/
//
// TODO: Store and process the castling rights.
// This struct loosely follows the format of the training data used by lc0.
//
// https://lczero.org/dev/wiki/training-data-format-versions/
#[derive(Debug)]
struct TrainingData {
    planes: Vec<u64>,
    best_q: f32,
    best_d: f32,
    best_idx: u16,
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
        assert_eq!(version, 6);
        let _input_format = reader.read_u32::<LittleEndian>()?;
        let mut _probabilities = vec![0.0; 1858];
        for prob in _probabilities.iter_mut() {
            *prob = reader.read_f32::<LittleEndian>()?;
        }

        let mut planes = vec![0; 104];
        for plane in planes.iter_mut() {
            *plane = reverse_bits_in_bytes(reader.read_u64::<LittleEndian>()?);
        }

        let _castling_us_ooo = reader.read_u8()?;
        let _castling_us_oo = reader.read_u8()?;
        let _castling_them_ooo = reader.read_u8()?;
        let _castling_them_oo = reader.read_u8()?;
        let _side_to_move_or_enpassant = reader.read_u8()?;
        let _rule50_count = reader.read_u8()?;
        let _invariance_info = reader.read_u8()?;
        let _dummy = reader.read_u8()?;

        let _root_q = reader.read_f32::<LittleEndian>()?;
        let best_q = reader.read_f32::<LittleEndian>()?;

        let _root_d = reader.read_f32::<LittleEndian>()?;
        let best_d = reader.read_f32::<LittleEndian>()?;

        let _root_m = reader.read_f32::<LittleEndian>()?;
        let _best_m = reader.read_f32::<LittleEndian>()?;
        let _plies_left = reader.read_f32::<LittleEndian>()?;
        let _result_q = reader.read_f32::<LittleEndian>()?;
        let _result_d = reader.read_f32::<LittleEndian>()?;
        let _played_q = reader.read_f32::<LittleEndian>()?;
        let _played_d = reader.read_f32::<LittleEndian>()?;
        let _played_m = reader.read_f32::<LittleEndian>()?;
        let _orig_q = reader.read_f32::<LittleEndian>()?;
        let _orig_d = reader.read_f32::<LittleEndian>()?;
        let _orig_m = reader.read_f32::<LittleEndian>()?;
        let _visits = reader.read_u32::<LittleEndian>()?;
        let _played_idx = reader.read_u16::<LittleEndian>()?;
        let best_idx = reader.read_u16::<LittleEndian>()?;
        let _policy_kld = reader.read_f32::<LittleEndian>()?;
        let _reserved = reader.read_u32::<LittleEndian>()?;

        Ok(TrainingData {
            planes,
            best_q,
            best_d,
            best_idx,
        })
    }

    fn to_position(&self) -> Board {
        Board::from_bitboards(
            ByRole {
                pawn: Bitboard(self.planes[0] | self.planes[6]),
                knight: Bitboard(self.planes[1] | self.planes[7]),
                bishop: Bitboard(self.planes[2] | self.planes[8]),
                rook: Bitboard(self.planes[3] | self.planes[9]),
                queen: Bitboard(self.planes[4] | self.planes[10]),
                king: Bitboard(self.planes[5] | self.planes[11]),
            },
            ByColor {
                white: Bitboard(self.planes[0..6].iter().fold(0, |acc, &x| acc | x)),
                black: Bitboard(self.planes[6..12].iter().fold(0, |acc, &x| acc | x)),
            },
        )
    }
}

fn process_position(data: TrainingData) {
    // Filter out positions with too few pieces that will be covered by Syzygy endgame tablebase.
    let pieces = data
        .planes
        .iter()
        .take(12)
        .fold(0, |acc, plane| acc + plane.count_ones());

    // Filter out positions with too few pieces.
    if pieces <= MIN_PIECES {
        return;
    }

    // Filter out promotions early.
    if preprocessing::IDX_TO_MOVE[data.best_idx as usize].len() > 4 {
        return;
    }

    let board = data.to_position();
    println!(
        "{} {:.3} {:.3} {}",
        board,
        data.best_q,
        data.best_d,
        preprocessing::IDX_TO_MOVE[data.best_idx as usize]
    );

    // TODO: Filter out captures.
    // TODO: Filter out checks.
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
