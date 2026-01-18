use anyhow::{bail, Context, Result};
use clap::{ArgGroup, Parser};
use opencv::{
    core::{self, Ptr, Rect, Scalar, Size},
    highgui, imgproc, objdetect,
    prelude::*,
    tracking::{TrackerKCF, TrackerKCF_Params},
    videoio,
};
use serde::Serialize;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser, Debug)]
#[command(name = "vjcount", about = "Viola-Jones human detection and counting")]
#[command(group(ArgGroup::new("input").required(true).args(["file", "rtsp"])))]
struct Args {
    #[arg(long, value_name = "PATH", conflicts_with = "rtsp")]
    file: Option<PathBuf>,
    #[arg(long, value_name = "URL", conflicts_with = "file")]
    rtsp: Option<String>,
    #[arg(long, default_value = "assets/cascades")]
    cascade_dir: PathBuf,
    #[arg(long, default_value_t = 1.1)]
    scale_factor: f64,
    #[arg(long, default_value_t = 6)]
    min_neighbors: i32,
    #[arg(long, default_value_t = 48)]
    min_size: i32,
    #[arg(long, default_value_t = 0.4)]
    nms_iou: f32,
    #[arg(long, default_value_t = 45)]
    max_missing: u32,
    #[arg(long, default_value_t = 0.0)]
    exclude_top_percent: f32,
    #[arg(long, default_value_t = 1.2)]
    min_aspect_ratio: f32,
    #[arg(long, default_value_t = 4.0)]
    max_aspect_ratio: f32,
    #[arg(long)]
    headless: bool,
    #[arg(long)]
    log_json: Option<PathBuf>,
    #[arg(long, default_value_t = 5)]
    log_interval_seconds: u64,
    /// Run detection every N frames (1 = every frame, 3 = every 3rd frame)
    #[arg(long, default_value_t = 3)]
    detection_interval: u64,
}

#[derive(Clone, Debug)]
struct Track {
    id: usize,
    rect: Rect,
}

/// State for a single tracked object using OpenCV KCF tracker (faster than CSRT)
struct TrackedObject {
    id: usize,
    tracker: Ptr<TrackerKCF>,
    rect: Rect,
    missing: u32,
}

#[derive(Debug, Default)]
struct TrackingStats {
    matched: usize,
    new_tracks: usize,
    unmatched_tracks: usize,
    active_tracks: usize,
}

/// Multi-object tracker using OpenCV KCF (Kernelized Correlation Filter)
/// for each tracked object. KCF is 3-5x faster than CSRT with good accuracy.
struct OpenCVTracker {
    tracks: HashMap<usize, TrackedObject>,
    next_id: usize,
    max_missing: u32,
    iou_threshold: f32,
}

impl OpenCVTracker {
    fn new(max_missing: u32, iou_threshold: f32) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
            max_missing,
            iou_threshold,
        }
    }

    fn total_unique(&self) -> usize {
        self.next_id.saturating_sub(1)
    }

    fn visible_tracks(&self) -> Vec<Track> {
        let mut tracks: Vec<Track> = self
            .tracks
            .values()
            .filter(|t| t.missing == 0)
            .map(|t| Track { id: t.id, rect: t.rect })
            .collect();
        tracks.sort_by_key(|track| track.id);
        tracks
    }

    fn update(&mut self, frame: &Mat, detections: &[Rect]) -> TrackingStats {
        let mut stats = TrackingStats::default();
        
        // Phase 1: Update only visible trackers with KCF prediction (skip missing ones for speed)
        let track_ids: Vec<usize> = self.tracks.keys().copied().collect();
        let mut predicted_rects: HashMap<usize, Option<Rect>> = HashMap::new();
        
        for track_id in &track_ids {
            if let Some(tracked) = self.tracks.get_mut(track_id) {
                // Skip tracker update for missing objects (performance optimization)
                if tracked.missing > 0 {
                    predicted_rects.insert(*track_id, Some(tracked.rect));
                    continue;
                }
                let mut new_rect = tracked.rect;
                match tracked.tracker.update(frame, &mut new_rect) {
                    Ok(true) => {
                        // Tracker successfully predicted new position
                        predicted_rects.insert(*track_id, Some(new_rect));
                    }
                    _ => {
                        // Tracker lost the target
                        predicted_rects.insert(*track_id, None);
                    }
                }
            }
        }
        
        // Phase 2: Match predictions to detections using IoU
        let mut matched_tracks: HashSet<usize> = HashSet::new();
        let mut matched_detections: HashSet<usize> = HashSet::new();
        
        // Build IoU pairs between predictions and detections
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (track_id, pred_rect_opt) in &predicted_rects {
            if let Some(pred_rect) = pred_rect_opt {
                for (det_idx, det_rect) in detections.iter().enumerate() {
                    let iou = rect_iou(*pred_rect, *det_rect);
                    if iou > 0.0 {
                        pairs.push((iou, *track_id, det_idx));
                    }
                }
            }
        }
        
        // Sort by IoU descending (greedy matching)
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        
        // Greedy assignment
        for (iou, track_id, det_idx) in pairs {
            if matched_tracks.contains(&track_id) || matched_detections.contains(&det_idx) {
                continue;
            }
            if iou >= self.iou_threshold {
                // Match found: use detection rect (more accurate than prediction)
                if let Some(tracked) = self.tracks.get_mut(&track_id) {
                    let det_rect = detections[det_idx];
                    tracked.rect = det_rect;
                    tracked.missing = 0;
                    // NOTE: Skipping tracker re-initialization for performance.
                    // The tracker will naturally drift toward the correct position.
                }
                matched_tracks.insert(track_id);
                matched_detections.insert(det_idx);
                stats.matched += 1;
            }
        }
        
        // Phase 3: Handle unmatched tracks
        for track_id in &track_ids {
            if matched_tracks.contains(track_id) {
                continue;
            }
            if let Some(tracked) = self.tracks.get_mut(track_id) {
                // If tracker had a valid prediction, use it but mark as missing
                if let Some(Some(pred_rect)) = predicted_rects.get(track_id) {
                    tracked.rect = *pred_rect;
                }
                tracked.missing = tracked.missing.saturating_add(1);
            }
        }
        
        // Phase 4: Create new tracks for unmatched detections
        for (det_idx, det_rect) in detections.iter().enumerate() {
            if matched_detections.contains(&det_idx) {
                continue;
            }
            if let Ok(mut new_tracker) = create_kcf_tracker() {
                if new_tracker.init(frame, *det_rect).is_ok() {
                    let tracked = TrackedObject {
                        id: self.next_id,
                        tracker: new_tracker,
                        rect: *det_rect,
                        missing: 0,
                    };
                    self.tracks.insert(self.next_id, tracked);
                    self.next_id += 1;
                    stats.new_tracks += 1;
                }
            }
        }
        
        // Phase 5: Remove tracks that have been missing too long
        let to_remove: Vec<usize> = self
            .tracks
            .iter()
            .filter_map(|(track_id, tracked)| {
                if tracked.missing > self.max_missing {
                    Some(*track_id)
                } else {
                    None
                }
            })
            .collect();
        for track_id in to_remove {
            self.tracks.remove(&track_id);
        }
        
        stats.unmatched_tracks = self.tracks.len().saturating_sub(matched_tracks.len());
        stats.active_tracks = self.tracks.len();
        stats
    }
}

/// Create a new KCF tracker instance with default parameters (faster than CSRT)
fn create_kcf_tracker() -> Result<Ptr<TrackerKCF>> {
    let params = TrackerKCF_Params::default()?;
    TrackerKCF::create(params).context("Failed to create KCF tracker")
}


#[derive(Debug, Default)]
struct MetricsAccumulator {
    tp: u64,
    fn_count: u64,
    frames: u64,
    detections: u64,
}

impl MetricsAccumulator {
    fn update(&mut self, detections: usize, tp: u64, fn_count: u64) {
        self.tp += tp;
        self.fn_count += fn_count;
        self.frames += 1;
        self.detections += detections as u64;
    }

    fn reset(&mut self) {
        *self = MetricsAccumulator::default();
    }

    fn detection_rate(&self) -> f64 {
        let denom = self.tp + self.fn_count;
        if denom == 0 {
            0.0
        } else {
            self.tp as f64 / denom as f64
        }
    }
}

#[derive(Serialize)]
struct SessionLog {
    event: &'static str,
    timestamp: String,
    source: String,
    cascade_dir: String,
    scale_factor: f64,
    min_neighbors: i32,
    min_size: i32,
    nms_iou: f32,
    max_missing: u32,
    iou_threshold: f32,
}

#[derive(Serialize)]
struct FrameLog {
    event: &'static str,
    timestamp: String,
    frame_index: u64,
    detections: usize,
    matched_tracks: usize,
    new_tracks: usize,
    unmatched_tracks: usize,
    active_tracks: usize,
    total_unique: usize,
    tp_total: u64,
    fn_total: u64,
    detection_rate: f64,
    detection_rate_percent: f64,
}

#[derive(Serialize)]
struct SummaryLog {
    event: &'static str,
    timestamp: String,
    frame_index: u64,
    interval_seconds: u64,
    interval_frames: u64,
    interval_detections: u64,
    total_unique: usize,
    tp_total: u64,
    fn_total: u64,
    detection_rate: f64,
    detection_rate_percent: f64,
}

struct JsonLogger {
    writer: BufWriter<File>,
}

impl JsonLogger {
    fn new(path: &Path) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    fn write_event<T: Serialize>(&mut self, event: &T) -> Result<()> {
        serde_json::to_writer(&mut self.writer, event)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();
    let args = Args::parse();
    run(args)
}

fn run(args: Args) -> Result<()> {
    let source = if let Some(file) = &args.file {
        file.to_string_lossy().to_string()
    } else if let Some(rtsp) = &args.rtsp {
        rtsp.clone()
    } else {
        bail!("Provide --file or --rtsp");
    };

    let fullbody_path = args.cascade_dir.join("haarcascade_fullbody.xml");
    let upperbody_path = args.cascade_dir.join("haarcascade_upperbody.xml");
    ensure_file_exists(&fullbody_path)?;
    ensure_file_exists(&upperbody_path)?;

    let mut fullbody = objdetect::CascadeClassifier::new(
        fullbody_path
            .to_str()
            .context("Fullbody cascade path is invalid")?,
    )
    .context("Failed to load fullbody cascade")?;
    let mut upperbody = objdetect::CascadeClassifier::new(
        upperbody_path
            .to_str()
            .context("Upperbody cascade path is invalid")?,
    )
    .context("Failed to load upperbody cascade")?;

    let mut capture =
        videoio::VideoCapture::from_file(&source, videoio::CAP_ANY).with_context(|| {
            format!("Failed to open input source: {}", source)
        })?;
    if !capture.is_opened()? {
        bail!("Failed to open input source: {}", source);
    }
    let _ = capture.set(videoio::CAP_PROP_BUFFERSIZE, 1.0);

    let mut json_logger = match args.log_json.as_ref() {
        Some(path) => Some(JsonLogger::new(path)?),
        None => None,
    };

    if let Some(logger) = json_logger.as_mut() {
        let session = SessionLog {
            event: "session_start",
            timestamp: timestamp_now(),
            source: source.clone(),
            cascade_dir: args.cascade_dir.display().to_string(),
            scale_factor: args.scale_factor,
            min_neighbors: args.min_neighbors,
            min_size: args.min_size,
            nms_iou: args.nms_iou,
            max_missing: args.max_missing,
            iou_threshold: args.nms_iou, // Using same IoU threshold for tracking
        };
        logger.write_event(&session)?;
        logger.flush()?;
    }

    let mut display_enabled = !args.headless;
    let window_name = "vjcount";
    if display_enabled {
        if let Err(err) = highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE) {
            tracing::warn!("Failed to open display window: {}. Running headless.", err);
            display_enabled = false;
        }
    }

    let mut tracker = OpenCVTracker::new(args.max_missing, args.nms_iou);
    let mut metrics = MetricsAccumulator::default();
    let mut interval_metrics = MetricsAccumulator::default();

    let start_time = Instant::now();
    let mut last_summary = Instant::now();
    let mut frame_index: u64 = 0;

    let mut frame = Mat::default();
    let mut gray = Mat::default();
    let mut gray_eq = Mat::default();
    let mut rects_full = core::Vector::<Rect>::new();
    let mut rects_upper = core::Vector::<Rect>::new();

    loop {
        if !capture.read(&mut frame)? {
            break;
        }
        if frame.empty() {
            break;
        }
        frame_index += 1;

        imgproc::cvt_color(
            &frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .context("Failed to convert to grayscale")?;
        imgproc::equalize_hist(&gray, &mut gray_eq)
            .context("Failed to equalize histogram")?;

        // Run detection only every N frames for performance (configurable via --detection-interval)
        let run_detection = frame_index % args.detection_interval == 0;
        
        // Skip expensive cascade detection entirely on non-detection frames
        if run_detection {
            rects_full.clear();
            rects_upper.clear();
            fullbody.detect_multi_scale(
                &gray_eq,
                &mut rects_full,
                args.scale_factor,
                args.min_neighbors,
                0,
                Size::new(args.min_size, args.min_size),
                Size::default(),
            )?;
            upperbody.detect_multi_scale(
                &gray_eq,
                &mut rects_upper,
                args.scale_factor,
                args.min_neighbors,
                0,
                Size::new(args.min_size, args.min_size),
                Size::default(),
            )?;
        }
        
        let detections = if run_detection {
            let mut detections: Vec<Rect> = rects_full.to_vec();
            detections.extend(rects_upper.to_vec());
            let detections = nms_rects(&detections, args.nms_iou);
            let frame_height = frame.rows();
            filter_detections(
                &detections,
                frame_height,
                args.exclude_top_percent,
                args.min_aspect_ratio,
                args.max_aspect_ratio,
            )
        } else {
            // On non-detection frames, just update trackers with empty detections
            Vec::new()
        };

        let stats = tracker.update(&frame, &detections);
        let tp_frame = (stats.matched + stats.new_tracks) as u64;
        let fn_frame = stats.unmatched_tracks as u64;

        metrics.update(detections.len(), tp_frame, fn_frame);
        interval_metrics.update(detections.len(), tp_frame, fn_frame);

        let detection_rate = metrics.detection_rate();
        let frame_log = FrameLog {
            event: "frame",
            timestamp: timestamp_now(),
            frame_index,
            detections: detections.len(),
            matched_tracks: stats.matched,
            new_tracks: stats.new_tracks,
            unmatched_tracks: stats.unmatched_tracks,
            active_tracks: stats.active_tracks,
            total_unique: tracker.total_unique(),
            tp_total: metrics.tp,
            fn_total: metrics.fn_count,
            detection_rate,
            detection_rate_percent: detection_rate * 100.0,
        };
        if let Some(logger) = json_logger.as_mut() {
            logger.write_event(&frame_log)?;
        }

        if display_enabled {
            draw_tracks(&mut frame, &tracker.visible_tracks())?;
            draw_hud(
                &mut frame,
                tracker.total_unique(),
                stats.active_tracks,
                start_time,
                frame_index,
            )?;
            highgui::imshow(window_name, &frame)?;
            let key = highgui::wait_key(1)?;
            if key == 27 || key == 113 {
                break;
            }
        }

        if last_summary.elapsed().as_secs() >= args.log_interval_seconds {
            let detection_rate = metrics.detection_rate();
            let summary = SummaryLog {
                event: "summary",
                timestamp: timestamp_now(),
                frame_index,
                interval_seconds: args.log_interval_seconds,
                interval_frames: interval_metrics.frames,
                interval_detections: interval_metrics.detections,
                total_unique: tracker.total_unique(),
                tp_total: metrics.tp,
                fn_total: metrics.fn_count,
                detection_rate,
                detection_rate_percent: detection_rate * 100.0,
            };
            tracing::info!(
                "frames={}, unique={}, detections={}, detection_rate={:.2}%",
                frame_index,
                tracker.total_unique(),
                interval_metrics.detections,
                detection_rate * 100.0
            );
            if let Some(logger) = json_logger.as_mut() {
                logger.write_event(&summary)?;
                logger.flush()?;
            }
            interval_metrics.reset();
            last_summary = Instant::now();
        }
    }

    if let Some(logger) = json_logger.as_mut() {
        logger.flush()?;
    }
    Ok(())
}

fn ensure_file_exists(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!("Required cascade file missing: {}", path.display());
    }
    Ok(())
}



fn filter_detections(
    rects: &[Rect],
    frame_height: i32,
    exclude_top_percent: f32,
    min_aspect_ratio: f32,
    max_aspect_ratio: f32,
) -> Vec<Rect> {
    let exclude_y_threshold = (frame_height as f32 * exclude_top_percent) as i32;
    
    rects
        .iter()
        .filter(|rect| {
            // ROI filter: exclude ceiling area
            if rect.y < exclude_y_threshold {
                return false;
            }
            
            // Aspect ratio filter: people are taller than wide
            let aspect_ratio = rect.height as f32 / rect.width.max(1) as f32;
            if aspect_ratio < min_aspect_ratio || aspect_ratio > max_aspect_ratio {
                return false;
            }
            
            true
        })
        .copied()
        .collect()
}

fn rect_area(rect: Rect) -> f32 {
    let area = rect.width.max(0) * rect.height.max(0);
    area as f32
}

fn rect_iou(a: Rect, b: Rect) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);

    let inter_w = (x2 - x1).max(0) as f32;
    let inter_h = (y2 - y1).max(0) as f32;
    let inter_area = inter_w * inter_h;

    let union = rect_area(a) + rect_area(b) - inter_area;
    if union <= 0.0 {
        0.0
    } else {
        inter_area / union
    }
}

fn nms_rects(rects: &[Rect], iou_threshold: f32) -> Vec<Rect> {
    if rects.is_empty() {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..rects.len()).collect();
    indices.sort_by(|&a, &b| {
        rect_area(rects[b])
            .partial_cmp(&rect_area(rects[a]))
            .unwrap_or(Ordering::Equal)
    });

    let mut keep: Vec<Rect> = Vec::new();
    for idx in indices {
        let rect = rects[idx];
        if keep
            .iter()
            .all(|kept| rect_iou(rect, *kept) <= iou_threshold)
        {
            keep.push(rect);
        }
    }
    keep
}

fn draw_tracks(frame: &mut Mat, tracks: &[Track]) -> Result<()> {
    let color = Scalar::new(0.0, 255.0, 0.0, 0.0);
    for track in tracks {
        imgproc::rectangle(frame, track.rect, color, 2, imgproc::LINE_8, 0)?;
        let label = format!("ID {}", track.id);
        let origin = core::Point::new(track.rect.x, track.rect.y.saturating_sub(6));
        imgproc::put_text(
            frame,
            &label,
            origin,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            imgproc::LINE_8,
            false,
        )?;
    }
    Ok(())
}

fn draw_hud(
    frame: &mut Mat,
    total_unique: usize,
    active_tracks: usize,
    start_time: Instant,
    frame_index: u64,
) -> Result<()> {
    let elapsed = start_time.elapsed().as_secs_f64();
    let fps = if elapsed > 0.0 {
        frame_index as f64 / elapsed
    } else {
        0.0
    };

    let color = Scalar::new(0.0, 255.0, 0.0, 0.0);
    let text = format!(
        "Unique: {} | Active: {} | FPS: {:.1}",
        total_unique, active_tracks, fps
    );
    imgproc::put_text(
        frame,
        &text,
        core::Point::new(10, 24),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        imgproc::LINE_8,
        false,
    )?;
    Ok(())
}

fn timestamp_now() -> String {
    chrono::Utc::now().to_rfc3339()
}
