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
use ordered_float::OrderedFloat;
use std::f64::consts::PI;

fn trs_to_mat(pos: DVec2, rot: f64, scale: f64) -> DMat3 {
    DMat3::from_scale_angle_translation(DVec2::splat(scale), rot, pos)
}

fn mat_to_trs(mat: &DMat3) -> (DVec2, f64, f64) {
    let pos = mat.z_axis.xy();
    let x_axis = mat.x_axis.xy();
    let scale = x_axis.length();
    let rot = x_axis.y.atan2(x_axis.x);
    (pos, rot, scale)
}

fn set_pos(mat: &mut DMat3, pos: DVec2) {
    let (_, rot, scale) = mat_to_trs(mat);
    *mat = trs_to_mat(pos, rot, scale);
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Copy, Debug, PartialEq)]
pub enum PortalType {
    Regular { scale_y: f64 },
    Perspective { scale_y: f64 },
    Wormhole { scale_y: f64 },
    Flat,
    Semicircle { scale_y: f64 },
}

impl Default for PortalType {
    fn default() -> Self {
        PortalType::Regular { scale_y: 1.0 }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone)]
pub struct Portal {
    portal1: DMat3,
    portal2: DMat3,
    kind: PortalType,
}

impl Default for Portal {
    fn default() -> Self {
        Self {
            portal1: trs_to_mat(DVec2::new(-1.01, 0.), 0.0, 1.0),
            portal2: trs_to_mat(DVec2::new(1.01, 0.), 0.0, 1.0),
            kind: Default::default(),
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
    fn new(o: DVec2, d: DVec2) -> Ray {
        Ray { o, d }
    }

    fn offset(&self, t: f64) -> DVec2 {
        self.o + self.d * t
    }

    fn normalize(&self) -> Ray {
        Ray {
            o: self.o,
            d: self.d.normalize(),
        }
    }

    fn transform(&self, matrix: &DMat3) -> Ray {
        Ray {
            o: matrix.transform_point2(self.o),
            d: matrix.transform_vector2(self.d),
        }
    }
}

fn intersect_ellipse(ray: &Ray, scale_y: f64) -> Option<f64> {
    if scale_y <= 0.0 {
        return None;
    }

    let o = DVec2::new(ray.o.x, ray.o.y / scale_y);
    let d = DVec2::new(ray.d.x, ray.d.y / scale_y);

    let a = d.dot(d);
    if a <= 1e-24 {
        return None;
    }
    let b = 2.0 * o.dot(d);
    let c = o.dot(o) - 1.0;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();

    let q = -0.5 * (b + b.signum() * sqrt_disc);

    let mut t0 = q / a;
    let mut t1 = c / q;
    if t0 > t1 {
        std::mem::swap(&mut t0, &mut t1);
    }

    const EPS: f64 = 1e-12;
    if t0 >= EPS {
        Some(t0)
    } else if t1 >= EPS {
        Some(t1)
    } else {
        None
    }
}

fn ray_circle_intersection(ray: &Ray) -> Option<f64> {
    let a = ray.d.dot(ray.d);
    let b = 2.0 * ray.o.dot(ray.d);
    let c = ray.o.dot(ray.o) - 1.0;
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

fn ray_segment_intersection(ray: &Ray) -> Option<f64> {
    if ray.d.y.abs() <= 1e-12 {
        return None;
    }
    let t = -ray.o.y / ray.d.y;
    if t < 1e-12 {
        return None;
    }
    let x = ray.o.x + t * ray.d.x;
    if (-1.0..=1.0).contains(&x) {
        Some(t)
    } else {
        None
    }
}

fn ray_semicircle_intersection(ray: &Ray) -> Option<f64> {
    let a = ray.d.dot(ray.d);
    let b = 2.0 * ray.o.dot(ray.d);
    let c = ray.o.dot(ray.o) - 1.0;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    let mut res: Option<f64> = None;
    for &t in [t1, t2].iter() {
        if t >= 1e-12 {
            let p = ray.offset(t);
            if p.y >= -1e-12 {
                res = match res {
                    Some(current) if current <= t => Some(current),
                    _ => Some(t),
                };
            }
        }
    }
    res
}

fn intersect_semicircle(ray: &Ray, scale_y: f64) -> Option<f64> {
    if scale_y == 1.0 {
        return ray_semicircle_intersection(ray);
    }
    if scale_y <= 0.0 {
        return None;
    }
    let scaled_ray = Ray {
        o: DVec2::new(ray.o.x, ray.o.y / scale_y),
        d: DVec2::new(ray.d.x, ray.d.y / scale_y),
    };
    ray_semicircle_intersection(&scaled_ray)
}

fn teleport_position(pos: DVec2, from: &DMat3, to: &DMat3) -> DVec2 {
    let local = from.inverse().transform_point2(pos);
    to.transform_point2(local)
}

fn teleport_direction(dir: DVec2, from: &DMat3, to: &DMat3) -> DVec2 {
    let local = from.inverse().transform_vector2(dir);
    to.transform_vector2(local)
}

fn circle_invert_ray_direction(ray: &Ray) -> DVec2 {
    let p = ray.o;
    let d = ray.d;
    let r2 = p.dot(p);
    if r2 == 0.0 {
        return d;
    }
    let dot = p.dot(d);
    let num = d * r2 - p * (2.0 * dot);
    let denom = r2 * r2;
    let inv = num / denom;
    let orig_len = d.length();
    let inv_len = inv.length();
    if inv_len == 0.0 {
        DVec2::ZERO
    } else {
        inv * (orig_len / inv_len)
    }
}

fn ellipse_invert_ray_direction(ray: &Ray, scale_y: f64) -> DVec2 {
    if scale_y <= 0.0 {
        return ray.d;
    }

    let p = ray.o;
    let nx = p.x;
    let ny = p.y / (scale_y * scale_y);

    let mut n = DVec2::new(nx, ny);
    let n_len = n.length();
    if n_len == 0.0 {
        return ray.d;
    }
    n /= n_len;

    let dot = ray.d.dot(n);
    ray.d - n * (2.0 * dot)
}

fn intersect_portal(ray: &Ray, portal: &Portal) -> Option<(f64, Ray)> {
    let inv1 = portal.portal1.inverse();
    let inv2 = portal.portal2.inverse();

    let local1 = ray.transform(&inv1);
    let local2 = ray.transform(&inv2);

    let (t1, t2) = match portal.kind {
        PortalType::Flat => (
            ray_segment_intersection(&local1),
            ray_segment_intersection(&local2),
        ),
        PortalType::Semicircle { scale_y } => (
            intersect_semicircle(&local1, scale_y),
            intersect_semicircle(&local2, scale_y),
        ),
        PortalType::Regular { scale_y }
        | PortalType::Perspective { scale_y }
        | PortalType::Wormhole { scale_y } => (
            if scale_y == 1.0 {
                ray_circle_intersection(&local1)
            } else {
                intersect_ellipse(&local1, scale_y)
            },
            if scale_y == 1.0 {
                ray_circle_intersection(&local2)
            } else {
                intersect_ellipse(&local2, scale_y)
            },
        ),
    };

    let (t_hit, from, to, inv_to) = match (t1, t2) {
        (Some(t1), Some(t2)) => {
            if t1 < t2 {
                (t1, &portal.portal1, &portal.portal2, &inv2)
            } else {
                (t2, &portal.portal2, &portal.portal1, &inv1)
            }
        }
        (Some(t1), None) => (t1, &portal.portal1, &portal.portal2, &inv2),
        (None, Some(t2)) => (t2, &portal.portal2, &portal.portal1, &inv1),
        (None, None) => return None,
    };

    let new_pos = teleport_position(ray.offset(t_hit), from, to);
    let mut new_dir = teleport_direction(ray.d, from, to);

    match portal.kind {
        PortalType::Wormhole { scale_y } => {
            let local_ray = Ray::new(new_pos, new_dir).transform(inv_to);
            let inverted = if scale_y == 1.0 {
                circle_invert_ray_direction(&local_ray)
            } else {
                ellipse_invert_ray_direction(&local_ray, scale_y)
            };
            new_dir = to.transform_vector2(inverted);
        }
        PortalType::Perspective { .. } => {
            new_dir = -new_dir;
        }
        PortalType::Regular { .. } | PortalType::Semicircle { .. } | PortalType::Flat => {}
    }

    Some((t_hit, Ray::new(new_pos, new_dir)))
}

fn travel_ray(
    ray: &Ray,
    portals: &[Portal],
    settings: &RenderingSettings,
) -> Vec<(DVec2, DVec2, f64)> {
    let mut ray = ray.normalize();
    let mut positions = std::collections::VecDeque::new();
    positions.push_back(ray.o);
    let mut lengths = Vec::new();

    for _ in 0..settings.max_teleportations {
        let res = portals
            .iter()
            .flat_map(|portal| intersect_portal(&ray, portal))
            .min_by_key(|(t, _)| OrderedFloat(*t));

        if let Some((t_val, mut teleported_ray)) = res {
            ray.o = ray.offset(t_val);
            positions.push_back(ray.o);
            lengths.push(ray.d.length());

            teleported_ray.o = teleported_ray.offset(settings.ray_offset);
            ray = teleported_ray;

            positions.push_back(ray.o);
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
            line_thickness: 0.01,
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

    fn get_matrices(&self, ui: &mut Ui) -> (DMat3, DMat3, DMat3) {
        let available_size = ui.max_rect().size();
        let window_scale = available_size.x.min(available_size.y);
        let mat_orig = self.get_matrix();
        let screen_mat =
            DMat3::from_translation(
                DVec2::new(available_size.x as f64, available_size.y as f64) / 2.,
            ) * DMat3::from_scale(DVec2::new(window_scale as f64, window_scale as f64));
        let mat = screen_mat * mat_orig;

        (mat, screen_mat, mat_orig)
    }

    fn update(&mut self, ui: &mut Ui, mat: DMat3, screen_mat: DMat3, mat_orig: DMat3) {
        let mut response = ui.allocate_rect(egui::Rect::EVERYTHING, egui::Sense::drag());
        if response.dragged() {
            ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::Move);
            let delta = response.drag_delta();
            let delta =
                (screen_mat.inverse() * DVec3::new(delta.x as f64, delta.y as f64, 0.)).xy();
            self.pos += delta;
            response.mark_changed();
        }
        let scale = ui
            .ctx()
            .input(|r| 1.003_f64.powf(r.smooth_scroll_delta.y as f64) * r.zoom_delta() as f64);
        if response.hovered() && scale != 1. {
            if let Some(pos) = response.hover_pos() {
                let pos = (mat.inverse() * DVec3::new(pos.x as f64, pos.y as f64, 1.)).xy();
                let mat1 = mat_orig
                    * DMat3::from_translation(pos)
                    * DMat3::from_scale(DVec2::new(scale, scale))
                    * DMat3::from_translation(-pos);
                self.pos = (mat1 * DVec3::new(0., 0., 1.)).xy();
                self.scale = (mat1 * DVec3::new(1., 0., 0.)).x;
            }
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Default)]
#[serde(default)]
pub struct Portals2D {
    portals: Vec<Portal>,
    ray: Ray,
    segment: SegmentLight,
    cone: ConeLight,
    camera: Camera,

    rendering_settings: RenderingSettings,

    drawing_settings: DrawingSettings,
}

impl Portals2D {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // if let Some(storage) = cc.storage {
        //     return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        // }

        let mut res: Portals2D = Default::default();
        res.portals.push(Default::default());
        res.portals.push(Default::default());
        res
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

fn draw_portal_shape(
    painter: &PainterWrapper<'_>,
    mat: &DMat3,
    kind: PortalType,
    thickness: f64,
    color: Color32,
) {
    let stroke = Stroke::new(thickness as f32, color);
    match kind {
        PortalType::Flat => {
            let a = mat.transform_point2(DVec2::new(-1.0, 0.0));
            let b = mat.transform_point2(DVec2::new(1.0, 0.0));
            painter.line_segment(a, b, stroke);
        }
        PortalType::Semicircle { scale_y } => {
            let segments = 32;
            let mut prev = mat.transform_point2(DVec2::new(-1.0, 0.0));
            for i in 1..=segments {
                let angle = PI - PI * i as f64 / segments as f64;
                let pt_local = DVec2::new(angle.cos(), angle.sin() * scale_y);
                let pt = mat.transform_point2(pt_local);
                painter.line_segment(prev, pt, stroke);
                prev = pt;
            }
        }
        PortalType::Regular { scale_y }
        | PortalType::Perspective { scale_y }
        | PortalType::Wormhole { scale_y } => {
            if scale_y == 1.0 {
                let (pos, _, r) = mat_to_trs(mat);
                painter.circle_stroke(pos, r - thickness / 2.0, stroke);
            } else {
                let segments = 64;
                let mut prev = mat.transform_point2(DVec2::new(1.0, 0.0));
                for i in 1..=segments {
                    let angle = 2.0 * PI * i as f64 / segments as f64;
                    let pt_local = DVec2::new(angle.cos(), angle.sin() * scale_y);
                    let pt = mat.transform_point2(pt_local);
                    painter.line_segment(prev, pt, stroke);
                    prev = pt;
                }
            }
        }
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
        let stroke = Stroke::new(drawing_settings.line_thickness as f32 * *len as f32, color);

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
                let mut stroke =
                    Stroke::new(drawing_settings.line_thickness as f32 * len as f32, color);

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

                        stroke =
                            Stroke::new(drawing_settings.line_thickness as f32 * len as f32, color);
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

                let (mat, screen_mat, mat_orig) = self.camera.get_matrices(ui);

                let painter = PainterWrapper::new(&painter, mat);

                let stroke = Stroke::new(
                    self.drawing_settings.portal_thickness as f32,
                    Color32::YELLOW,
                );
                painter.line_segment(self.segment.start, self.segment.end, stroke);

                let rays = segment_perpendicular_points(&self.segment);
                for ray in rays {
                    let segments = travel_ray(&ray, &self.portals, &self.rendering_settings);
                    draw_segments(&painter, &segments, &self.drawing_settings);
                }

                let rays = position_angular_rays(&self.cone);
                for ray in rays {
                    let segments = travel_ray(&ray, &self.portals, &self.rendering_settings);
                    draw_segments(&painter, &segments, &self.drawing_settings);
                }

                let segments = travel_ray(&self.ray, &self.portals, &self.rendering_settings);
                draw_segments(&painter, &segments, &self.drawing_settings);

                for portal in &mut self.portals {
                    let (mut pos1, _, _) = mat_to_trs(&portal.portal1);
                    draw_portal_shape(
                        &painter,
                        &portal.portal1,
                        portal.kind,
                        self.drawing_settings.portal_thickness,
                        Color32::ORANGE,
                    );
                    point_ui_mat(ui, mat, &mut pos1, 1., 1.);
                    set_pos(&mut portal.portal1, pos1);

                    let (mut pos2, _, _) = mat_to_trs(&portal.portal2);
                    draw_portal_shape(
                        &painter,
                        &portal.portal2,
                        portal.kind,
                        self.drawing_settings.portal_thickness,
                        Color32::LIGHT_BLUE,
                    );
                    point_ui_mat(ui, mat, &mut pos2, 1., 1.);
                    set_pos(&mut portal.portal2, pos2);
                }

                point_direction_ui_mat(ui, mat, self.ray.o, &mut self.ray.d, 1., 1.);
                point_ui_mat(ui, mat, &mut self.ray.o, 1., 1.);

                point_ui_mat(ui, mat, &mut self.segment.start, 1., 1.);
                point_ui_mat(ui, mat, &mut self.segment.end, 1., 1.);

                point_ui_mat(ui, mat, &mut self.cone.position, 1., 1.);

                self.camera.update(ui, mat, screen_mat, mat_orig);
            });

        egui::Window::new("Parameters")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                if ui.button("Reset").clicked() {
                    ui.memory_mut(|mem| *mem = Default::default());
                    *self = Default::default();
                }
                ui.separator();
                for portal in &mut self.portals {
                    ui.horizontal(|ui| {
                        ui.label("Portal 1:");
                        let (mut pos, mut rot, mut r) = mat_to_trs(&portal.portal1);
                        let mut changed = false;
                        changed |= egui_f64(ui, &mut pos.x);
                        changed |= egui_f64(ui, &mut pos.y);
                        ui.separator();
                        changed |= egui_angle_f64(ui, &mut rot);
                        ui.separator();
                        changed |= egui_f64_positive(ui, &mut r);
                        if changed {
                            portal.portal1 = trs_to_mat(pos, rot, r);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Portal 2:");
                        let (mut pos, mut rot, mut r) = mat_to_trs(&portal.portal2);
                        let mut changed = false;
                        changed |= egui_f64(ui, &mut pos.x);
                        changed |= egui_f64(ui, &mut pos.y);
                        ui.separator();
                        changed |= egui_angle_f64(ui, &mut rot);
                        ui.separator();
                        changed |= egui_f64_positive(ui, &mut r);
                        if changed {
                            portal.portal2 = trs_to_mat(pos, rot, r);
                        }
                    });
                    let scale_y = match portal.kind {
                        PortalType::Regular { scale_y }
                        | PortalType::Perspective { scale_y }
                        | PortalType::Wormhole { scale_y }
                        | PortalType::Semicircle { scale_y } => scale_y,
                        _ => 1.0,
                    };
                    ui.horizontal(|ui| {
                        ui.label("Portal type:");
                        ui.selectable_value(
                            &mut portal.kind,
                            PortalType::Regular { scale_y },
                            "Regular",
                        );
                        ui.selectable_value(
                            &mut portal.kind,
                            PortalType::Wormhole { scale_y },
                            "Wormhole",
                        );
                        ui.selectable_value(
                            &mut portal.kind,
                            PortalType::Perspective { scale_y },
                            "Perspective",
                        );
                        ui.selectable_value(&mut portal.kind, PortalType::Flat, "Flat");
                        ui.selectable_value(
                            &mut portal.kind,
                            PortalType::Semicircle { scale_y },
                            "Semicircle",
                        );
                    });
                    if let PortalType::Regular { scale_y }
                    | PortalType::Wormhole { scale_y }
                    | PortalType::Perspective { scale_y }
                    | PortalType::Semicircle { scale_y } = &mut portal.kind
                    {
                        ui.horizontal(|ui| {
                            ui.label("Scale Y:");
                            egui_f64_positive(ui, scale_y);
                        });
                    }
                }
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
