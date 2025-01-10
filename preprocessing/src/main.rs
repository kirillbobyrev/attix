use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use flate2::read::GzDecoder;
use shakmaty::{Bitboard, Board, ByColor, ByRole};
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

// Positions with small number of pieces are usually adjudicated by Syzygy endgame tablebases.
const MIN_PIECES: u32 = 7;

// Each plane is a distinct bitboard representing a piece type of a certain color.
const NUM_PLANES: usize = 12;

// A position from the training data with accompanying metadata.
//
// Original format: https://lczero.org/dev/wiki/training-data-format-versions/
#[derive(Debug)]
struct TrainingSample {
    bitboards: [u64; NUM_PLANES],
    // Prediction targets.
    best_q: f32,
    best_d: f32,
    castling_us_ooo: bool,
    castling_us_oo: bool,
    castling_them_ooo: bool,
    castling_them_oo: bool,
    // Index of the best move in the policy head. See preprocessing::IDX_TO_MOVE.
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

impl TrainingSample {
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

        let castling_us_ooo = reader.read_u8()? != 0;
        let castling_us_oo = reader.read_u8()? != 0;
        let castling_them_ooo = reader.read_u8()? != 0;
        let castling_them_oo = reader.read_u8()? != 0;
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

        Ok(TrainingSample {
            bitboards: planes[0..NUM_PLANES].try_into().unwrap(),
            best_q,
            best_d,
            best_idx,
            castling_us_ooo,
            castling_us_oo,
            castling_them_ooo,
            castling_them_oo,
        })
    }

    fn to_board(&self) -> Board {
        Board::from_bitboards(
            ByRole {
                pawn: Bitboard(self.bitboards[0] | self.bitboards[6]),
                knight: Bitboard(self.bitboards[1] | self.bitboards[7]),
                bishop: Bitboard(self.bitboards[2] | self.bitboards[8]),
                rook: Bitboard(self.bitboards[3] | self.bitboards[9]),
                queen: Bitboard(self.bitboards[4] | self.bitboards[10]),
                king: Bitboard(self.bitboards[5] | self.bitboards[11]),
            },
            ByColor {
                white: Bitboard(self.bitboards[0..6].iter().fold(0, |acc, &x| acc | x)),
                black: Bitboard(
                    self.bitboards[6..NUM_PLANES]
                        .iter()
                        .fold(0, |acc, &x| acc | x),
                ),
            },
        )
    }
}

struct CastlingBitboards {
    castling_us_oo: u64,
    castling_us_ooo: u64,
    castling_them_oo: u64,
    castling_them_ooo: u64,
}

fn process_position(data: TrainingSample, castling: &CastlingBitboards) {
    // Filter out positions with too few pieces that will be covered by Syzygy endgame tablebase.
    let num_pieces = data
        .bitboards
        .iter()
        .fold(0, |acc, plane| acc + plane.count_ones());
    if num_pieces <= MIN_PIECES {
        return;
    }

    // Filter out promotions early.
    if preprocessing::IDX_TO_MOVE[data.best_idx as usize].len() > 4 {
        return;
    }

    let board = data.to_board();
    // println!(
    //     "{} {:.3} {:.3} {} {} {} {} {}",
    //     board,
    //     data.best_q,
    //     data.best_d,
    //     preprocessing::IDX_TO_MOVE[data.best_idx as usize],
    //     data.castling_us_ooo,
    //     data.castling_us_oo,
    //     data.castling_them_ooo,
    //     data.castling_them_oo,
    // );

    // TODO: Filter out captures.
    // TODO: Filter out checks.
}

fn process_game<R: Read>(reader: R) -> io::Result<()> {
    let mut gz = GzDecoder::new(reader);

    // The first position in the game has rooks placed on the castling squares.
    let initial_position = TrainingSample::read_from(&mut gz)?;
    let initial_board = initial_position.to_board();

    // Calculate the bitboards for castling (initial rook positions).
    let our_rooks = initial_position.bitboards[3];
    let castling_us_oo_bitboard = our_rooks & (!our_rooks + 1);
    let castling_us_ooo_bitboard = our_rooks ^ castling_us_oo_bitboard;
    let castling_them_oo_bitboard = castling_us_oo_bitboard.swap_bytes();
    let castling_them_ooo_bitboard = castling_us_ooo_bitboard.swap_bytes();

    let castling_bitboards = CastlingBitboards {
        castling_us_oo: castling_us_oo_bitboard,
        castling_us_ooo: castling_us_ooo_bitboard,
        castling_them_oo: castling_them_oo_bitboard,
        castling_them_ooo: castling_them_ooo_bitboard,
    };

    println!(
        "{} {} {} {} {}",
        initial_board,
        castling_us_oo_bitboard,
        castling_us_ooo_bitboard,
        castling_them_oo_bitboard,
        castling_them_ooo_bitboard
    );

    // TODO: lc0 training data does not contain en passant squares, but those
    // can be retroactively calculated.
    while let Ok(data) = TrainingSample::read_from(&mut gz) {
        process_position(data, &castling_bitboards);
        break;
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

        process_game(entry)?;
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    process_tar_file(&args.tar_path)
}
