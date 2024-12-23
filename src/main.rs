use arrow::array::{ArrayRef, UInt32Builder};
use arrow::compute::{concat_batches, take};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::{RecordBatch, RecordBatchReader};

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use parquet::schema::printer::print_schema;

use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::fs::{create_dir_all, read_dir, File};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use uuid::Uuid;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use clap::Parser;

/// Configuration for chunked shuffle
// #[derive(Clone)]
// struct ShuffleConfig {
//     pub chunk_size: usize,        // how many rows to accumulate before shuffling
//     pub batch_size: usize,        // how many rows to read at a time (per RecordBatch)
//     pub output_file_rows: usize,  // how many rows per output parquet file
//     pub threads: usize,           // how many threads total
// }

/// Simple program to download a webpage and extract its text content
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The input file: main parameter
    #[arg(short, long, required = true)]
    input: String,

    /// The batch size for parallel processing (default is 8192)
    #[arg(short, long, default_value_t = 4)]
    num_workers: usize,

    /// The output file: main parameter
    #[arg(short, long, required = true)]
    output: String,

    /// Shuffle size
    #[arg(short, long, default_value_t = 500_000)]
    shuffle_size: usize,

    /// Output file number of rows
    #[arg(short, long, default_value_t = 64_000)]
    fragment_size: usize,
}

fn main() -> ArrowResult<()> {
    let args: Args = Args::parse();
    let input_dir = args.input;
    let output_dir = Arc::new(args.output);
    let num_workers = args.num_workers;
    let fragment_size = args.fragment_size;
    let shuffle_size = args.shuffle_size;


    // Ensure the output directory exists
    create_dir_all(output_dir.to_string()).expect("Unable to create output directory");

    // You can use rayon's thread-pool builder or standard threads.
    // For demonstration, let's just spawn standard threads in `multi_thread_shuffle`.
    let multi_progress = MultiProgress::new();

    // Main progress bar
    let progress_bar = multi_progress.add(ProgressBar::new_spinner());
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {pos} ({percent}%): {per_sec}")
            .expect("Failed to set progress bar style")
            .progress_chars("#>-"),
    );
    let progress_bar = Arc::new(progress_bar);

    // Logging progress bar (dedicated for error/output messages)
    let log_bar = Arc::new(multi_progress.add(ProgressBar::new_spinner()));
    log_bar.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.red} {msg}")
            .expect("Failed to set log bar style"),
    );

    // Collect all .parquet files
    let mut parquet_paths: Vec<PathBuf> = read_dir(&input_dir)
        .unwrap_or_else(|_| panic!("Cannot read input directory {}", input_dir))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().map_or(false, |ext| ext == "parquet") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    parquet_paths.sort(); // optional sort for deterministic ordering

    // Then shuffle the parquet files names
    let mut rng = rand::thread_rng();
    parquet_paths.shuffle(&mut rng);

    let num_files = parquet_paths.len();
    if num_files == 0 {
        eprintln!("No parquet files found in {}", &input_dir);
        return Ok(()); // nothing to do
    }

    // If there are fewer files than threads, clamp the thread count
    let n_threads = num_workers.min(num_files).max(1);

    // Divide the files into roughly equal subsets
    let chunk_size = (num_files + n_threads - 1) / n_threads; // round-up division
    let mut subsets = Vec::new();
    let mut start_idx = 0;
    for _ in 0..n_threads {
        let end_idx = (start_idx + chunk_size).min(num_files);
        if start_idx >= end_idx {
            break;
        }
        let subset = parquet_paths[start_idx..end_idx].to_vec();
        subsets.push(subset);
        start_idx = end_idx;
    }

    // Spawn a thread per subset
    let mut handles = Vec::new();
    for subset in subsets {
        let pb = Arc::clone(&progress_bar);
        let output_dir = Arc::clone(&output_dir);
        let out_dir = output_dir.to_string();
        let handle = thread::spawn(move || {
            // Each thread processes its subset of files independently
            if let Err(e) = shuffle_subset(&subset, shuffle_size, fragment_size, &out_dir, pb) {
                eprintln!("Thread error: {:#?}", e);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for h in handles {
        h.join().expect("Failed to join thread");
    }

    Ok(())
}


/// Shuffle a subset of files with chunked logic (read -> accumulate -> shuffle -> write).
fn shuffle_subset(paths: &[PathBuf], shuffle_size: usize, fragment_size: usize, output_dir: &str, progress_bar: Arc<ProgressBar>) -> ArrowResult<()> {
    // Accumulate up to config.chunk_size rows
    let mut accumulator: Vec<RecordBatch> = Vec::new();
    let mut accumulator_count = 0usize;

    for path in paths {
        read_parquet_in_batches(path, shuffle_size - accumulator_count, shuffle_size, &mut |batch| {
            accumulator_count += batch.num_rows();
            progress_bar.inc(batch.num_rows() as u64);
            accumulator.push(batch);

            let schema = accumulator[0].schema();

            // If we reached chunk_size, shuffle & write
            if accumulator_count >= shuffle_size {
                assert_eq!(shuffle_size, accumulator_count);
                let concatenated = concat_batches(&schema, &accumulator)?;
                let shuffled = shuffle_table(&concatenated)?;
                write_in_splits(&shuffled, fragment_size, output_dir)?;
                accumulator.clear();
                accumulator_count = 0;
            }
            Ok(())
        })?;
    }

    // Write any leftover rows
    if !accumulator.is_empty() {
        let schema = accumulator[0].schema();
        let concatenated = concat_batches(&schema, &accumulator)?;
        let shuffled = shuffle_table(&concatenated)?;
        write_in_splits(&shuffled, fragment_size, output_dir)?;
    }

    Ok(())
}

/// Read a parquet file in `batch_size` increments using ParquetRecordBatchReaderBuilder.
/// Since we read multiple batches, maybe a previous batch was not large enough and was missing
/// a few rows to reach the shuffle batch_size. This is why we have a `first_batch_size` parameter,
/// that controls the size of the first batch read from the file (if there any enough rows). The rest of the batches will be
/// read with the `batch_size` parameter.
fn read_parquet_in_batches<F>(
    path: &Path,
    first_batch_size: usize,
    batch_size: usize,
    on_batch: &mut F,
) -> ArrowResult<()>
where
    F: FnMut(RecordBatch) -> ArrowResult<()>,
{
    let file = File::open(path)?;
    let first_builder = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(first_batch_size);

    let mut reader = first_builder.build()?;

    // Iterate through RecordBatches
    if let Some(batch_result) = reader.next() {
        let batch = batch_result?;
        let has_not_enough_rows = batch.num_rows() <= first_batch_size;
        on_batch(batch)?;

        if has_not_enough_rows {
            return Ok(());
        }
    }

    let file = File::open(path)?;
    let mut batch_builder = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(batch_size)
        .with_offset(first_batch_size);

    let mut reader = batch_builder.build()?;
    while let Some(batch_result) = reader.next() {
        let batch = batch_result?;
        on_batch(batch)?;
    }
    Ok(())
}

/// Randomly shuffle the rows in a RecordBatch.
fn shuffle_table(batch: &RecordBatch) -> ArrowResult<RecordBatch> {
    let n_rows = batch.num_rows();
    if n_rows == 0 {
        return Ok(batch.clone());
    }

    // Generate random permutation of indices [0..n_rows)
    let mut indices: Vec<u32> = (0..n_rows as u32).collect();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    // Convert indices to an Arrow array
    let mut builder = UInt32Builder::new();
    for idx in indices {
        builder.append_value(idx);
    }
    let idx_array = Arc::new(builder.finish());

    // "take" each column using the shuffled indices
    let mut shuffled_columns = Vec::new();
    for col in batch.columns() {
        let taken = take(col.as_ref(), &idx_array, None)?;
        shuffled_columns.push(taken);
    }

    // Construct a new RecordBatch with the same schema
    RecordBatch::try_new(batch.schema(), shuffled_columns)
}

/// Write the `batch` in splits of size `rows_per_file` each, with random UUID filenames.
fn write_in_splits(batch: &RecordBatch, rows_per_file: usize, output_dir: &str) -> ArrowResult<()> {
    let n_rows = batch.num_rows();
    let mut offset = 0;

    while offset < n_rows {
        let end = std::cmp::min(offset + rows_per_file, n_rows);
        let slice = batch.slice(offset, end - offset);

        // Generate a random UUID for the output file name
        let file_name = format!("part-{}.parquet", Uuid::new_v4());
        let file_path = Path::new(output_dir).join(file_name);

        write_parquet(&slice, &file_path)?;
        offset = end;
    }
    Ok(())
}

/// Write a single RecordBatch to a Parquet file (single row group).
fn write_parquet(batch: &RecordBatch, path: &Path) -> ArrowResult<()> {
    use parquet::arrow::arrow_writer::ArrowWriter;

    let file = File::create(path)?;
    // Customize compression, encoding, etc.
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::SNAPPY)
        .build();

    // Create a Parquet writer from the Arrow schema
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(batch)?;
    writer.close()?;

    Ok(())
}