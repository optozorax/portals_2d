use egui::Color32;
use egui::DragValue;
use egui::Painter;
use egui::Sense;
use egui::Stroke;
use egui::Ui;
use egui::Vec2;
use glam::swizzles::Vec3Swizzles;
use glam::DMat3;
use glam::DVec2;
use glam::DVec3;
use std::f64::consts::PI;

#[derive(serde::Deserialize, serde::Serialize, Clone)]
pub struct Circle {
    pos: DVec2,
    rot: f64,
    r: f64,
}

impl Default for Circle {
    fn default() -> Self {
        Circle {
            pos: DVec2::ZERO,
            rot: 0.,
            r: 1.,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Default, Clone, Copy, Debug, PartialEq)]
pub enum PortalType {
    #[default]
    Regular,
    Perspective,
    Wormhole,
}

#[derive(serde::Deserialize, serde::Serialize, Clone)]
pub struct Portal {
    c1: Circle,
    c2: Circle,
    portal_type: PortalType,
}

impl Default for Portal {
    fn default() -> Self {
        Self {
            c1: Circle {
                pos: DVec2::new(-1.01, 0.),
                rot: 0.,
                r: 1.,
            },
            c2: Circle {
                pos: DVec2::new(1.01, 0.),
                rot: 0.,
                r: 1.,
            },
            portal_type: Default::default(),
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone)]
pub struct Ray {
    o: DVec2,
    d: DVec2,
}

impl Default for Ray {
    fn default() -> Self {
        Self {
            o: DVec2::new(-2.5, 0.5),
            d: DVec2::new(0.2, 0.),
        }
    }
}

impl Ray {
    fn offset(&self, t: f64) -> DVec2 {
        self.o + self.d * t
    }

    fn normalize(&self) -> Ray {
        Ray {
            o: self.o,
            d: self.d.normalize(),
        }
    }
}

fn ray_circle_intersection(ray: &Ray, circle: &Circle) -> Option<f64> {
    let oc = ray.o - circle.pos;

    let a = ray.d.dot(ray.d);
    let b = 2.0 * oc.dot(ray.d);
    let c = oc.dot(oc) - circle.r * circle.r;

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None;
    }

    let sqrt_discriminant = discriminant.sqrt();
    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
    let t2 = (-b + sqrt_discriminant) / (2.0 * a);

    if t1 >= 0.0 && t2 >= 0.0 {
        Some(t1.min(t2))
    } else if t1 >= 0.0 {
        Some(t1)
    } else if t2 >= 0.0 {
        Some(t2)
    } else {
        None
    }
}

fn teleport_position(pos: DVec2, from: &Circle, to: &Circle) -> DVec2 {
    let x_local = pos.x - from.pos.x;
    let y_local = pos.y - from.pos.y;

    let cos_r1 = (-from.rot).cos();
    let sin_r1 = (-from.rot).sin();
    let x_rot = x_local * cos_r1 - y_local * sin_r1;
    let y_rot = x_local * sin_r1 + y_local * cos_r1;

    let x_local_final = x_rot / from.r;
    let y_local_final = y_rot / from.r;

    let x_scaled = x_local_final * to.r;
    let y_scaled = y_local_final * to.r;

    let cos_r2 = to.rot.cos();
    let sin_r2 = to.rot.sin();
    let x_rot2 = x_scaled * cos_r2 - y_scaled * sin_r2;
    let y_rot2 = x_scaled * sin_r2 + y_scaled * cos_r2;

    let x_final = x_rot2 + to.pos.x;
    let y_final = y_rot2 + to.pos.y;

    DVec2::new(x_final, y_final)
}

fn teleport_direction(dir: DVec2, from: &Circle, to: &Circle) -> DVec2 {
    let cos_r1 = (-from.rot).cos();
    let sin_r1 = (-from.rot).sin();
    let x_rot = dir.x * cos_r1 - dir.y * sin_r1;
    let y_rot = dir.x * sin_r1 + dir.y * cos_r1;

    let x_local = x_rot / from.r;
    let y_local = y_rot / from.r;

    let x_scaled = x_local * to.r;
    let y_scaled = y_local * to.r;

    let cos_r2 = to.rot.cos();
    let sin_r2 = to.rot.sin();
    let x_final = x_scaled * cos_r2 - y_scaled * sin_r2;
    let y_final = x_scaled * sin_r2 + y_scaled * cos_r2;

    DVec2::new(x_final, y_final)
}

fn circle_invert_ray_direction(ray: &Ray, circle: &Circle) -> DVec2 {
    let p = ray.o - circle.pos;
    let d = ray.d;

    let r2 = p.dot(p);
    if r2 == 0.0 {
        return d;
    }

    let dot = p.dot(d);

    let num = d * r2 - p * (2.0 * dot);
    let denom = r2 * r2;

    let inv = num * (circle.r * circle.r) / denom;

    let orig_len = d.length();
    let inv_len = inv.length();
    if inv_len == 0.0 {
        return DVec2::ZERO;
    }

    let scale = orig_len / inv_len;
    inv * scale
}

fn intersect_circle_portal(ray: &Ray, portal: &Portal) -> (Option<f64>, bool, Option<Ray>) {
    let t1 = ray_circle_intersection(ray, &portal.c1);
    let t2 = ray_circle_intersection(ray, &portal.c2);

    if let Some(t1_val) = t1 {
        if t2.is_none() || t1_val < t2.unwrap() {
            let new_pos = teleport_position(ray.offset(t1_val), &portal.c1, &portal.c2);
            let mut new_dir = teleport_direction(ray.d, &portal.c1, &portal.c2);

            match portal.portal_type {
                PortalType::Wormhole => {
                    new_dir = circle_invert_ray_direction(
                        &Ray {
                            o: new_pos,
                            d: new_dir,
                        },
                        &portal.c2,
                    );
                }
                PortalType::Perspective => {
                    new_dir = -new_dir;
                }
                PortalType::Regular => {}
            }

            return (
                Some(t1_val),
                true,
                Some(Ray {
                    o: new_pos,
                    d: new_dir,
                }),
            );
        }
    }

    if let Some(t2_val) = t2 {
        if t1.is_none() || t2_val < t1.unwrap() {
            let new_pos = teleport_position(ray.offset(t2_val), &portal.c2, &portal.c1);
            let mut new_dir = teleport_direction(ray.d, &portal.c2, &portal.c1);

            match portal.portal_type {
                PortalType::Wormhole => {
                    new_dir = circle_invert_ray_direction(
                        &Ray {
                            o: new_pos,
                            d: new_dir,
                        },
                        &portal.c1,
                    );
                }
                PortalType::Perspective => {
                    new_dir = -new_dir;
                }
                PortalType::Regular => {}
            }

            return (
                Some(t2_val),
                true,
                Some(Ray {
                    o: new_pos,
                    d: new_dir,
                }),
            );
        }
    }

    (None, false, None)
}

fn travel_ray(
    ray: &Ray,
    portal: &Portal,
    settings: &RenderingSettings,
) -> Vec<(DVec2, DVec2, f64)> {
    let mut ray = ray.normalize();
    let mut positions = std::collections::VecDeque::new();
    positions.push_back(ray.o);
    let mut lengths = Vec::new();

    for _ in 0..settings.max_teleportations {
        let (t, is_continue, new_ray) = intersect_circle_portal(&ray, portal);

        if let Some(t_val) = t {
            ray.o = ray.offset(t_val);
            positions.push_back(ray.o);
            lengths.push(ray.d.length());

            if let Some(mut teleported_ray) = new_ray {
                teleported_ray.o = teleported_ray.offset(settings.ray_offset);
                ray = teleported_ray;

                if is_continue {
                    positions.push_back(ray.o);
                } else {
                    break;
                }
            }
        } else {
            ray.o = ray.offset(settings.end_offset);
            positions.push_back(ray.o);
            lengths.push(ray.d.length());
            break;
        }
    }

    let mut result = vec![];
    for len in lengths {
        let start = positions.pop_front().unwrap();
        let end = positions.pop_front().unwrap();
        result.push((start, end, len));
    }
    result
}

fn segment_perpendicular_points(segment: &SegmentLight) -> Vec<Ray> {
    if segment.count == 0 {
        return Vec::new();
    }

    let segment_vector = segment.end - segment.start;
    let mut perp_direction = DVec2::new(-segment_vector.y, segment_vector.x);

    let perp_length = perp_direction.length();
    if perp_length > 0.0 {
        perp_direction /= perp_length;
    } else {
        perp_direction = DVec2::new(0.0, 1.0);
    }

    let mut result = Vec::new();

    if segment.count == 1 {
        let origin = (segment.start + segment.end) / 2.0;
        result.push(Ray {
            o: origin,
            d: perp_direction,
        });
    } else {
        for i in 0..segment.count {
            let t = i as f64 / (segment.count - 1) as f64;
            let origin = segment.start + t * segment_vector;
            result.push(Ray {
                o: origin,
                d: perp_direction,
            });
        }
    }

    result
}

fn position_angular_rays(cone: &ConeLight) -> Vec<Ray> {
    if cone.count == 0 {
        return Vec::new();
    }

    let mut result = Vec::new();

    if cone.count == 1 {
        let direction = DVec2::new(cone.orientation.cos(), cone.orientation.sin());
        result.push(Ray {
            o: cone.position,
            d: direction,
        });
    } else {
        for i in 0..cone.count {
            let base_angle = i as f64 * cone.spread / (cone.count - 1) as f64;
            let final_angle = cone.orientation - cone.spread / 2. + base_angle;
            let direction = DVec2::new(final_angle.cos(), final_angle.sin());
            result.push(Ray {
                o: cone.position,
                d: direction,
            });
        }
    }

    result
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct SegmentLight {
    start: DVec2,
    end: DVec2,
    count: u32,
}

impl Default for SegmentLight {
    fn default() -> Self {
        Self {
            start: DVec2::new(1.5, 1.5),
            end: DVec2::new(0.5, 1.5),
            count: 20,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct ConeLight {
    position: DVec2,
    orientation: f64,
    spread: f64,
    count: u32,
}

impl Default for ConeLight {
    fn default() -> Self {
        Self {
            position: DVec2::new(-2.5, -0.5),
            orientation: 0.,
            spread: 0.2,
            count: 10,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
struct RenderingSettings {
    max_teleportations: usize,
    end_offset: f64,
    ray_offset: f64,
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            max_teleportations: 100,
            end_offset: 100.,
            ray_offset: 0.0001,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
struct DrawingSettings {
    line_thickness: f64,
    portal_thickness: f64,
    draw_after_position: bool,
    light_position: f64,
    light_radius: f64,

    trace_size: f64,
    trace_count: usize,
    draw_trace: bool,
}

impl Default for DrawingSettings {
    fn default() -> Self {
        Self {
            line_thickness: 0.005,
            portal_thickness: 0.01,
            draw_after_position: true,
            light_position: 1.5,
            light_radius: 0.01,

            trace_size: 1.,
            trace_count: 20,
            draw_trace: true,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
#[serde(default)]
struct Camera {
    pos: DVec2,
    scale: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pos: DVec2::ZERO,
            scale: 0.3,
        }
    }
}

impl Camera {
    fn get_matrix(&self) -> DMat3 {
        DMat3::from_scale_angle_translation(DVec2::new(self.scale, self.scale), 0., self.pos)
    }
}

#[derive(serde::Deserialize, serde::Serialize, Default)]
#[serde(default)]
pub struct Portals2D {
    portal: Portal,
    ray: Ray,
    segment: SegmentLight,
    cone: ConeLight,
    camera: Camera,

    rendering_settings: RenderingSettings,

    drawing_settings: DrawingSettings,
}

impl Portals2D {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Default::default()
    }
}

pub fn point_ui(ui: &mut egui::Ui, pos: &mut DVec2, coef_x: f64, coef_y: f64) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(1.0, 1.0);

    let rect = egui::Rect::from_min_size(
        egui::pos2((pos.x * coef_x) as f32, (pos.y * coef_y) as f32) - desired_size / 2.,
        desired_size,
    );
    let mut response = ui.allocate_rect(rect, egui::Sense::drag());

    if response.dragged() {
        ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::Move);
        let delta = response.drag_delta();
        pos.x += delta.x as f64 / coef_x;
        pos.y += delta.y as f64 / coef_y;
        response.mark_changed();
    }

    if ui.is_rect_visible(rect) {
        let visuals = ui
            .style()
            .interact_selectable(&response, response.dragged());
        let radius = 0.5 * rect.height();
        let rect = rect.expand(visuals.expansion);
        let center = egui::pos2(rect.center().x, rect.center().y);
        ui.painter()
            .circle(center, 0.75 * radius, visuals.bg_fill, visuals.fg_stroke);
    }

    response
}

pub fn point_ui_mat(
    ui: &mut egui::Ui,
    mat: DMat3,
    pos: &mut DVec2,
    coef_x: f64,
    coef_y: f64,
) -> egui::Response {
    *pos = (mat * DVec3::new(pos.x, pos.y, 1.)).xy();
    let res = point_ui(ui, pos, coef_x, coef_y);
    *pos = (mat.inverse() * DVec3::new(pos.x, pos.y, 1.)).xy();
    res
}

pub fn point_direction_ui(
    ui: &mut egui::Ui,
    pos: DVec2,
    dir: &mut DVec2,
    coef_x: f64,
    coef_y: f64,
) -> egui::Response {
    let mut pos2 = pos + *dir;
    let res = point_ui(ui, &mut pos2, coef_x, coef_y);
    *dir = pos2 - pos;
    res
}

pub fn point_direction_ui_mat(
    ui: &mut egui::Ui,
    mat: DMat3,
    mut pos: DVec2,
    dir: &mut DVec2,
    coef_x: f64,
    coef_y: f64,
) -> egui::Response {
    pos = (mat * DVec3::new(pos.x, pos.y, 1.)).xy();
    *dir = (mat * DVec3::new(dir.x, dir.y, 0.)).xy();
    let res = point_direction_ui(ui, pos, dir, coef_x, coef_y);
    *dir = (mat.inverse() * DVec3::new(dir.x, dir.y, 0.)).xy();
    res
}

fn glam_to_egui(pos: DVec2) -> egui::Pos2 {
    egui::Pos2::new(pos.x as f32, pos.y as f32)
}

pub fn check_changed<T: PartialEq + Clone, F: FnOnce(&mut T)>(t: &mut T, f: F) -> bool {
    let previous = t.clone();
    f(t);
    previous != *t
}

pub fn egui_angle_f64(ui: &mut Ui, angle: &mut f64) -> bool {
    let mut current = *angle / PI * 180.;
    let previous = current;
    ui.add(
        DragValue::from_get_set(|v| {
            if let Some(v) = v {
                if v > 360. {
                    current = v % 360.;
                } else if v < 0. {
                    current = 360. + (v % 360.);
                } else {
                    current = v;
                }
            }
            current
        })
        .speed(1)
        .suffix("°"),
    );
    if (previous - current).abs() > 1e-6 {
        *angle = current * PI / 180.;
        true
    } else {
        false
    }
}

pub fn egui_f64(ui: &mut Ui, value: &mut f64) -> bool {
    check_changed(value, |value| {
        ui.add(
            DragValue::new(value)
                .speed(0.01)
                .min_decimals(0)
                .max_decimals(2),
        );
    })
}

pub fn egui_f64_positive(ui: &mut Ui, value: &mut f64) -> bool {
    check_changed(value, |value| {
        ui.add(
            DragValue::new(value)
                .speed(0.01)
                .range(0.0..=1000.0)
                .min_decimals(0)
                .max_decimals(2),
        );
    })
}

struct PainterWrapper<'a> {
    painter: &'a Painter,
    transform: DMat3,
}

impl<'a> PainterWrapper<'a> {
    fn new(painter: &'a Painter, transform: DMat3) -> PainterWrapper<'a> {
        Self { painter, transform }
    }
}

impl PainterWrapper<'_> {
    fn transform_position(&self, a: DVec2) -> DVec2 {
        use glam::swizzles::Vec3Swizzles;
        (self.transform * glam::DVec3::new(a.x, a.y, 1.)).xy()
    }

    fn transform_distance(&self, d: f64) -> f64 {
        (self.transform * glam::DVec3::new(d, 0., 0.)).x
    }

    fn line_segment(&self, a: DVec2, b: DVec2, mut stroke: Stroke) {
        stroke.width *= self.transform_distance(1.) as f32;
        self.painter.line_segment(
            [
                glam_to_egui(self.transform_position(a)),
                glam_to_egui(self.transform_position(b)),
            ],
            stroke,
        );
    }

    fn circle_filled(&self, pos: DVec2, radius: f64, color: Color32) {
        self.painter.circle_filled(
            glam_to_egui(self.transform_position(pos)),
            self.transform_distance(radius) as f32,
            color,
        );
    }

    fn circle_stroke(&self, pos: DVec2, radius: f64, mut stroke: Stroke) {
        stroke.width *= self.transform_distance(1.) as f32;
        self.painter.circle_stroke(
            glam_to_egui(self.transform_position(pos)),
            self.transform_distance(radius) as f32,
            stroke,
        );
    }
}

fn draw_segments(
    painter: &PainterWrapper<'_>,
    segments: &[(DVec2, DVec2, f64)],
    drawing_settings: &DrawingSettings,
) {
    let mut circle: Option<(DVec2, f64)> = None;

    let color = Color32::WHITE.gamma_multiply(0.3);
    let mut pos = drawing_settings.light_position;
    for (start, end, len) in segments {
        let stroke = Stroke::new(
            (drawing_settings.line_thickness as f32 * *len as f32).clamp(0.01, 10.0),
            color,
        );

        let len1 = (end - start).length();
        let total_len = len1 / len;

        let t = pos / total_len;
        if 0. < t && t < 1. {
            let point = start + (end - start) * t;
            circle = Some((point, drawing_settings.light_radius * *len));
            if !drawing_settings.draw_after_position {
                painter.line_segment(*start, point, stroke);
                break;
            }
        }
        pos -= total_len;

        painter.line_segment(*start, *end, stroke);
    }

    let mut segment_iter = segments.iter();
    if drawing_settings.draw_trace {
        if let Some(res) = segment_iter.next() {
            let mut pos_trace = drawing_settings.light_position - drawing_settings.trace_size;
            let size_trace = drawing_settings.trace_size / drawing_settings.trace_count as f64;
            let (mut start, mut end, mut len) = *res;
            let mut len1 = (end - start).length();
            let mut total_len = len1 / len;
            'outer: for (_i, opacity) in (0..drawing_settings.trace_count)
                .map(|i| (i, i as f32 / drawing_settings.trace_count as f32))
            {
                let color = Color32::LIGHT_BLUE.gamma_multiply(opacity);
                let mut stroke = Stroke::new(
                    (drawing_settings.line_thickness as f32 * len as f32).clamp(0.01, 10.0),
                    color,
                );

                let mut t1 = pos_trace / total_len;
                let mut t2 = (pos_trace + size_trace) / total_len;

                loop {
                    if t1 <= 1. && t2 > 0. {
                        let seg_start = start + (end - start) * t1.clamp(0., 1.);
                        let seg_end = start + (end - start) * t2.clamp(0., 1.);
                        painter.line_segment(seg_start, seg_end, stroke);
                    }

                    if t2 > 1. {
                        pos_trace -= total_len;
                        (start, end, len) = if let Some(res) = segment_iter.next() {
                            *res
                        } else {
                            break 'outer;
                        };

                        len1 = (end - start).length();
                        total_len = len1 / len;

                        t1 = pos_trace / total_len;
                        t2 = (pos_trace + size_trace) / total_len;

                        stroke = Stroke::new(
                            (drawing_settings.line_thickness as f32 * len as f32).clamp(0.01, 10.0),
                            color,
                        );
                    } else {
                        break;
                    }
                }

                pos_trace += size_trace;
            }
        }
    }

    if let Some((pos, radius)) = circle {
        painter.circle_filled(pos, radius, Color32::YELLOW);
    }
}

impl eframe::App for Portals2D {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_theme(egui::Theme::Dark);

        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .inner_margin(0.)
                    .fill(ctx.style().visuals.panel_fill),
            )
            .show(ctx, |ui| {
                let (_, painter) = ui.allocate_painter(
                    Vec2::new(ui.available_width(), ui.available_height()),
                    Sense::hover(),
                );
                let available_size = ui.max_rect().size();
                let window_scale = available_size.x.min(available_size.y);
                let mat_orig = self.camera.get_matrix();
                let screen_mat =
                    DMat3::from_translation(
                        DVec2::new(available_size.x as f64, available_size.y as f64) / 2.,
                    ) * DMat3::from_scale(DVec2::new(window_scale as f64, window_scale as f64));
                let mat = screen_mat * self.camera.get_matrix();
                let painter = PainterWrapper::new(&painter, mat);

                let stroke = Stroke::new(
                    self.drawing_settings.portal_thickness as f32,
                    Color32::YELLOW,
                );
                painter.line_segment(self.segment.start, self.segment.end, stroke);

                let rays = segment_perpendicular_points(&self.segment);
                for ray in rays {
                    let segments = travel_ray(&ray, &self.portal, &self.rendering_settings);
                    draw_segments(&painter, &segments, &self.drawing_settings);
                }

                let rays = position_angular_rays(&self.cone);
                for ray in rays {
                    let segments = travel_ray(&ray, &self.portal, &self.rendering_settings);
                    draw_segments(&painter, &segments, &self.drawing_settings);
                }

                let segments = travel_ray(&self.ray, &self.portal, &self.rendering_settings);
                draw_segments(&painter, &segments, &self.drawing_settings);

                painter.circle_stroke(
                    self.portal.c1.pos,
                    self.portal.c1.r - self.drawing_settings.portal_thickness / 2.,
                    Stroke::new(
                        self.drawing_settings.portal_thickness as f32,
                        Color32::ORANGE,
                    ),
                );

                painter.circle_stroke(
                    self.portal.c2.pos,
                    self.portal.c2.r - self.drawing_settings.portal_thickness / 2.,
                    Stroke::new(
                        self.drawing_settings.portal_thickness as f32,
                        Color32::LIGHT_BLUE,
                    ),
                );

                point_ui_mat(ui, mat, &mut self.portal.c1.pos, 1., 1.);
                point_ui_mat(ui, mat, &mut self.portal.c2.pos, 1., 1.);

                point_direction_ui_mat(ui, mat, self.ray.o, &mut self.ray.d, 1., 1.);
                point_ui_mat(ui, mat, &mut self.ray.o, 1., 1.);

                point_ui_mat(ui, mat, &mut self.segment.start, 1., 1.);
                point_ui_mat(ui, mat, &mut self.segment.end, 1., 1.);

                point_ui_mat(ui, mat, &mut self.cone.position, 1., 1.);

                let mut response = ui.allocate_rect(egui::Rect::EVERYTHING, egui::Sense::drag());
                if response.dragged() {
                    ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::Move);
                    let delta = response.drag_delta();
                    let delta = (screen_mat.inverse()
                        * DVec3::new(delta.x as f64, delta.y as f64, 0.))
                    .xy();
                    self.camera.pos += delta;
                    response.mark_changed();
                }
                let wheel = ui.ctx().input(|r| r.smooth_scroll_delta);
                if response.hovered() && wheel.y != 0. {
                    if let Some(pos) = response.hover_pos() {
                        let pos = (mat.inverse() * DVec3::new(pos.x as f64, pos.y as f64, 1.)).xy();
                        let scale = 1.003_f64.powf(wheel.y as f64);
                        let mat1 = mat_orig
                            * DMat3::from_translation(pos)
                            * DMat3::from_scale(DVec2::new(scale, scale))
                            * DMat3::from_translation(-pos);
                        self.camera.pos = (mat1 * DVec3::new(0., 0., 1.)).xy();
                        self.camera.scale = (mat1 * DVec3::new(1., 0., 0.)).x;
                    }
                }
            });

        egui::Window::new("Parameters")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Portal 1:");
                    egui_f64(ui, &mut self.portal.c1.pos.x);
                    egui_f64(ui, &mut self.portal.c1.pos.y);
                    ui.separator();
                    egui_angle_f64(ui, &mut self.portal.c1.rot);
                    ui.separator();
                    egui_f64_positive(ui, &mut self.portal.c1.r);
                });
                ui.horizontal(|ui| {
                    ui.label("Portal 2:");
                    egui_f64(ui, &mut self.portal.c2.pos.x);
                    egui_f64(ui, &mut self.portal.c2.pos.y);
                    ui.separator();
                    egui_angle_f64(ui, &mut self.portal.c2.rot);
                    ui.separator();
                    egui_f64_positive(ui, &mut self.portal.c2.r);
                });
                ui.horizontal(|ui| {
                    ui.label("Portal type:");
                    ui.selectable_value(
                        &mut self.portal.portal_type,
                        PortalType::Regular,
                        "Regular",
                    );
                    ui.selectable_value(
                        &mut self.portal.portal_type,
                        PortalType::Wormhole,
                        "Wormhole",
                    );
                    ui.selectable_value(
                        &mut self.portal.portal_type,
                        PortalType::Perspective,
                        "Perspective",
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Ray:");
                    egui_f64(ui, &mut self.ray.o.x);
                    egui_f64(ui, &mut self.ray.o.y);
                    ui.separator();
                    egui_f64(ui, &mut self.ray.d.x);
                    egui_f64(ui, &mut self.ray.d.y);
                });
                ui.horizontal(|ui| {
                    ui.label("Segment:");
                    egui_f64(ui, &mut self.segment.start.x);
                    egui_f64(ui, &mut self.segment.start.y);
                    ui.separator();
                    egui_f64(ui, &mut self.segment.end.x);
                    egui_f64(ui, &mut self.segment.end.y);
                    ui.separator();
                    ui.add(DragValue::new(&mut self.segment.count));
                });
                ui.horizontal(|ui| {
                    ui.label("Cone:");
                    egui_f64(ui, &mut self.cone.position.x);
                    egui_f64(ui, &mut self.cone.position.y);
                    ui.separator();
                    egui_angle_f64(ui, &mut self.cone.orientation);
                    egui_angle_f64(ui, &mut self.cone.spread);
                    ui.separator();
                    ui.add(DragValue::new(&mut self.cone.count));
                });
                ui.separator();
                ui.label("Rendering settings");
                ui.horizontal(|ui| {
                    ui.label("Depth: ");
                    ui.add(DragValue::new(
                        &mut self.rendering_settings.max_teleportations,
                    ));
                });
                // todo: other
                ui.separator();
                ui.label("Drawing settings");
                ui.horizontal(|ui| {
                    ui.label("Line thickness: ");
                    egui_f64_positive(ui, &mut self.drawing_settings.line_thickness);
                });
                ui.horizontal(|ui| {
                    ui.label("Light position: ");
                    egui_f64_positive(ui, &mut self.drawing_settings.light_position);
                });
                ui.add(egui::Checkbox::new(
                    &mut self.drawing_settings.draw_after_position,
                    "Draw after position",
                ));
                ui.add(egui::Checkbox::new(
                    &mut self.drawing_settings.draw_trace,
                    "Draw trace",
                ));
                ui.horizontal(|ui| {
                    ui.label("Light radius: ");
                    egui_f64_positive(ui, &mut self.drawing_settings.light_radius);
                });
                ui.horizontal(|ui| {
                    ui.label("Trace size: ");
                    egui_f64_positive(ui, &mut self.drawing_settings.trace_size);
                });
                ui.horizontal(|ui| {
                    ui.label("Trace count: ");
                    ui.add(DragValue::new(&mut self.drawing_settings.trace_count));
                });
                ui.horizontal(|ui| {
                    ui.label("Portal thickness: ");
                    egui_f64_positive(ui, &mut self.drawing_settings.portal_thickness);
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Camera: ");
                    egui_f64(ui, &mut self.camera.pos.x);
                    egui_f64(ui, &mut self.camera.pos.y);
                    ui.separator();
                    ui.add(
                        DragValue::new(&mut self.camera.scale)
                            .speed(0.01)
                            .range(0.001..=1000.0)
                            .min_decimals(0)
                            .max_decimals(2),
                    );
                });
            });
    }
}
