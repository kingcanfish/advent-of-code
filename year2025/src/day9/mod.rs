use crate::file::read_file;

fn parse_input(content: &str) -> Vec<(i64, i64)> {
    content
        .lines()
        .map(|line| {
            let (a, b) = line.split_once(",").unwrap();
            (a.trim().parse().unwrap(), b.trim().parse().unwrap())
        })
        .collect()
}

fn bounding_area(p1: &(i64, i64), p2: &(i64, i64)) -> i64 {
    let dx = (p1.0 - p2.0).abs() + 1;
    let dy = (p1.1 - p2.1).abs() + 1;
    dx * dy
}

fn resolve_part1(points: &[(i64, i64)]) -> i64 {
    points
        .iter()
        .enumerate()
        .flat_map(|(i, p1)| points[i + 1..].iter().map(|p2| bounding_area(p1, p2)))
        .max()
        .unwrap_or(0)
}
// Point in polygon test using ray casting algorithm
// Cast a ray from point to the right and count intersections with polygon edges
fn point_in_polygon(point: (i64, i64), polygon: &[(i64, i64)]) -> bool {
    let (x, y) = point;
    let mut inside = false;
    let n = polygon.len();

    for i in 0..n {
        let (x1, y1) = polygon[i];
        let (x2, y2) = polygon[(i + 1) % n];

        // Check if the point is on a vertex
        if (x == x1 && y == y1) || (x == x2 && y == y2) {
            return true;
        }

        // Check if point is on a horizontal edge
        if y1 == y2 && y == y1 && x >= x1.min(x2) && x <= x1.max(x2) {
            return true;
        }

        // Check if point is on a vertical edge
        if x1 == x2 && x == x1 && y >= y1.min(y2) && y <= y1.max(y2) {
            return true;
        }

        // Ray casting: check if horizontal ray from point intersects edge
        if (y1 > y) != (y2 > y) {
            // For vertical edges
            if x1 == x2 {
                if x < x1 {
                    inside = !inside;
                }
            } else {
                // For non-vertical edges (shouldn't happen in rectilinear polygon)
                let x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
                if x < x_intersect {
                    inside = !inside;
                }
            }
        }
    }

    inside
}

// Check if a rectangle is entirely within the polygon
// For a rectilinear polygon, we check corners and some edge points
fn rectangle_in_polygon(p1: (i64, i64), p2: (i64, i64), polygon: &[(i64, i64)]) -> bool {
    let x1 = p1.0.min(p2.0);
    let x2 = p1.0.max(p2.0);
    let y1 = p1.1.min(p2.1);
    let y2 = p1.1.max(p2.1);

    // Check all four corners
    if !point_in_polygon((x1, y1), polygon) {
        return false;
    }
    if !point_in_polygon((x1, y2), polygon) {
        return false;
    }
    if !point_in_polygon((x2, y1), polygon) {
        return false;
    }
    if !point_in_polygon((x2, y2), polygon) {
        return false;
    }

    // For rectilinear polygons, if all corners are inside, we should also check
    // some points along the edges to handle non-convex cases
    // Sample points along edges
    let sample_count = 100.min((x2 - x1).max(y2 - y1)) as usize;

    // Check top and bottom edges
    if sample_count > 2 {
        for i in 1..sample_count {
            let x = x1 + (x2 - x1) * i as i64 / sample_count as i64;
            if !point_in_polygon((x, y1), polygon) {
                return false;
            }
            if !point_in_polygon((x, y2), polygon) {
                return false;
            }
        }

        // Check left and right edges
        for i in 1..sample_count {
            let y = y1 + (y2 - y1) * i as i64 / sample_count as i64;
            if !point_in_polygon((x1, y), polygon) {
                return false;
            }
            if !point_in_polygon((x2, y), polygon) {
                return false;
            }
        }
    }

    true
}

fn resolve_part2(points: &[(i64, i64)]) -> i64 {
    let mut max_area = 0;

    // Try all pairs of red tiles as opposite corners
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let p1 = points[i];
            let p2 = points[j];

            // Check if rectangle is within polygon
            if rectangle_in_polygon(p1, p2, points) {
                let area = bounding_area(&p1, &p2);
                max_area = max_area.max(area);
            }
        }
    }

    max_area
}

pub(crate) fn run() {
    let input = read_file("day9.txt").unwrap();
    let points = parse_input(&input);
    println!(
        "Day9 Part 1: {}, Part2: {}",
        resolve_part1(&points),
        resolve_part2(&points)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT: &str = r#"7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3"#;

    #[test]
    fn test_part1() {
        let points = parse_input(INPUT);
        assert_eq!(resolve_part1(&points), 50);
    }

    #[test]
    fn test_part2() {
        let points = parse_input(INPUT);
        assert_eq!(resolve_part2(&points), 24);
    }
}
