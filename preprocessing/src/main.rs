use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use flate2::read::GzDecoder;
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
    side_to_move_or_enpassant: u8,
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
            side_to_move_or_enpassant: reader.read_u8()?,
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

    fn print_board(&self) {
        println!("\nBoard Visualization:");
        println!("   a b c d e f g h");
        println!("   ---------------");

        // Define piece characters for both sides
        const OUR_PIECES: [char; 6] = ['P', 'N', 'B', 'R', 'Q', 'K'];
        const THEIR_PIECES: [char; 6] = ['p', 'n', 'b', 'r', 'q', 'k'];

        // Create a 8x8 board representation
        let mut board = vec![vec!['.'; 8]; 8];

        // Fill board with pieces
        for side in 0..2 {
            let pieces = if side == 0 {
                &OUR_PIECES
            } else {
                &THEIR_PIECES
            };
            for piece_type in 0..6 {
                let plane_idx = side * 6 + piece_type;
                let bitboard = self.planes[plane_idx];

                for square_idx in 0..64 {
                    if (bitboard >> square_idx) & 1 == 1 {
                        let rank = square_idx / 8;
                        let file = square_idx % 8;
                        board[7 - rank as usize][file as usize] = pieces[piece_type];
                    }
                }
            }
        }

        // Print the board
        for rank in 0..8 {
            // Iterate from 0 to 7 for correct indexing
            print!("{}  ", 8 - rank); // Print rank from 8 down to 1
            for file in 0..8 {
                print!("{} ", board[rank][file]);
            }
            println!(" {}", 8 - rank);
        }
        println!("   ---------------");
        println!("   a b c d e f g h");

        println!("\nCastling Rights:");
        println!(
            "Our side:   O-O: {}  O-O-O: {}",
            if self.castling_us_oo == 1 {
                "Yes"
            } else {
                "No"
            },
            if self.castling_us_ooo == 1 {
                "Yes"
            } else {
                "No"
            }
        );
        println!(
            "Their side: O-O: {}  O-O-O: {}",
            if self.castling_them_oo == 1 {
                "Yes"
            } else {
                "No"
            },
            if self.castling_them_ooo == 1 {
                "Yes"
            } else {
                "No"
            }
        );

        // Print side to move
        println!(
            "\nSide to move: {}",
            if self.side_to_move_or_enpassant == 1 {
                "Us"
            } else {
                "Them"
            }
        );

        println!("Rule 50: {}", self.rule50_count);
    }
}

fn process_gz_file<R: Read>(reader: R) -> io::Result<Vec<TrainingData>> {
    let mut gz = GzDecoder::new(reader);
    let mut training_data = Vec::new();

    loop {
        match TrainingData::read_from(&mut gz) {
            Ok(data) => training_data.push(data),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }

    Ok(training_data)
}

fn process_tar_file<P: AsRef<Path>>(path: P) -> io::Result<Vec<TrainingData>> {
    let file = File::open(path)?;
    let mut archive = Archive::new(file);
    let mut all_training_data = Vec::new();

    let mut idx: usize = 0;
    for entry in archive.entries()? {
        let entry = entry?;
        if !entry.path()?.to_string_lossy().ends_with(".gz") {
            continue;
        }

        match process_gz_file(entry) {
            Ok(mut data) => all_training_data.append(&mut data),
            Err(e) => eprintln!("Error processing gz file: {}", e),
        }
        idx += 1;
        if idx > 10 {
            break;
        }
    }

    Ok(all_training_data)
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    match process_tar_file(&args.tar_path) {
        Ok(training_data) => {
            println!(
                "Successfully processed {} training examples",
                training_data.len()
            );

            // Print details of specified number of entries
            for (i, data) in training_data.iter().take(args.num_entries).enumerate() {
                println!("Entry {}", i + 1);
                assert_eq!(data.version, 6);
                println!("Best Q: {}", data.best_q);
                println!("Best D: {}", data.best_d);
                println!("Best Index: {}", data.best_idx);

                data.print_board();

                println!("---");
            }
        }
        Err(e) => eprintln!("Error processing tar file: {}", e),
    }

    Ok(())
}
