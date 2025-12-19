use std::error::Error;
use std::path::PathBuf;

// Type alias for better readability
type ShapeMatrix = Vec<Vec<bool>>;

#[allow(unused)]
pub(crate) fn read_input() -> Result<Vec<String>, Box<dyn Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("inputs")
        .join("day12.txt");
    let lines: Vec<String> = std::fs::read_to_string(path)?
        .lines()
        .map(|s| s.to_string())
        .collect();
    Ok(lines)
}

/// Represents a shape with all its possible variants (rotations and flips)
#[derive(Clone)]
struct Shape {
    variants: Vec<ShapeMatrix>,
    area: usize,
}

/// Rotates a shape 90 degrees clockwise
fn rotate(shape: &[Vec<bool>]) -> ShapeMatrix {
    let n = shape.len();
    let mut rotated = vec![vec![false; n]; n];
    for i in 0..n {
        for j in 0..n {
            rotated[j][n - 1 - i] = shape[i][j];
        }
    }
    rotated
}

/// Flips a shape horizontally
fn flip(shape: &[Vec<bool>]) -> ShapeMatrix {
    let n = shape.len();
    let mut flipped = vec![vec![false; n]; n];
    for i in 0..n {
        for j in 0..n {
            flipped[i][n - 1 - j] = shape[i][j];
        }
    }
    flipped
}

/// Parses shape definitions from input lines
/// Generates all 8 variants (4 rotations + 4 rotations of flip) for each shape
fn parse_shapes(lines: &[String]) -> Vec<Shape> {
    const NUM_SHAPES: usize = 6;
    const SHAPE_HEIGHT: usize = 3;
    const NUM_ROTATIONS: usize = 4;

    let mut line_idx = 0;
    let mut shapes = Vec::with_capacity(NUM_SHAPES);

    for _ in 0..NUM_SHAPES {
        // Skip shape label (e.g., "i:")
        line_idx += 1;

        // Parse 3x3 shape matrix
        let mut shape = Vec::with_capacity(SHAPE_HEIGHT);
        for _ in 0..SHAPE_HEIGHT {
            let row: Vec<bool> = lines[line_idx].chars().map(|c| c == '#').collect();
            shape.push(row);
            line_idx += 1;
        }

        // Skip empty line
        line_idx += 1;

        // Calculate shape area
        let area = shape.iter().flatten().filter(|&&cell| cell).count();

        // Generate all 8 variants (4 rotations + 4 rotations of horizontal flip)
        let mut variants = Vec::with_capacity(8);

        // Add 4 rotations of original shape
        let mut current = shape.clone();
        for _ in 0..NUM_ROTATIONS {
            variants.push(current.clone());
            current = rotate(&current);
        }

        // Add 4 rotations of flipped shape
        current = flip(&shape);
        for _ in 0..NUM_ROTATIONS {
            variants.push(current.clone());
            current = rotate(&current);
        }

        shapes.push(Shape { variants, area });
    }

    shapes
}

/// Parses a region specification line (e.g., "5x3: 1 0 1 0 1 0")
/// Returns (width, height, counts) where counts is the number of each shape
fn parse_region(line: &str) -> (usize, usize, Vec<usize>) {
    let colon_pos = line.find(':').unwrap();
    let dimensions = &line[..colon_pos];
    let counts_str = &line[colon_pos + 2..];

    let x_pos = dimensions.find('x').unwrap();
    let width: usize = dimensions[..x_pos].parse().unwrap();
    let height: usize = dimensions[x_pos + 1..].parse().unwrap();

    let counts: Vec<usize> = counts_str
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();

    (width, height, counts)
}

/// Checks if all shapes can fit in the given grid dimensions
fn can_fit(shapes: &[Shape], width: usize, height: usize, counts: Vec<usize>) -> bool {
    // Build list of shape indices to place, expanding counts into individual shapes
    let shapes_to_place: Vec<usize> = counts
        .iter()
        .enumerate()
        .flat_map(|(shape_idx, &count)| vec![shape_idx; count])
        .collect();

    // Sort by area (largest first) for better backtracking performance
    let mut sorted_shapes = shapes_to_place;
    sorted_shapes.sort_by(|&a, &b| shapes[b].area.cmp(&shapes[a].area));

    let total_area: usize = sorted_shapes.iter().map(|&idx| shapes[idx].area).sum();
    let total_cells = width * height;
    let mut grid = vec![vec![false; width]; height];

    backtrack(
        &mut grid,
        shapes,
        &sorted_shapes,
        0,
        total_area,
        0,
        total_cells,
    )
}

/// Recursive backtracking to place shapes on the grid
fn backtrack(
    grid: &mut [Vec<bool>],
    shapes: &[Shape],
    shapes_to_place: &[usize],
    current_index: usize,
    remaining_area: usize,
    occupied: usize,
    total_cells: usize,
) -> bool {
    // Base case: all shapes placed successfully
    if current_index == shapes_to_place.len() {
        return true;
    }

    // Pruning: if remaining shapes can't fit in available space
    if occupied + remaining_area > total_cells {
        return false;
    }

    let shape_idx = shapes_to_place[current_index];
    let shape = &shapes[shape_idx];

    // Try all variants (rotations and flips) of the current shape
    for variant in &shape.variants {
        let variant_height = variant.len();
        let variant_width = variant[0].len();

        // Try all possible positions
        for x in 0..=grid.len().saturating_sub(variant_height) {
            for y in 0..=grid[0].len().saturating_sub(variant_width) {
                if can_place(grid, variant, x, y) {
                    place(grid, variant, x, y);

                    if backtrack(
                        grid,
                        shapes,
                        shapes_to_place,
                        current_index + 1,
                        remaining_area - shape.area,
                        occupied + shape.area,
                        total_cells,
                    ) {
                        return true;
                    }

                    unplace(grid, variant, x, y);
                }
            }
        }
    }

    false
}

/// Checks if a shape can be placed at position (x, y) without overlapping
fn can_place(grid: &[Vec<bool>], shape: &[Vec<bool>], x: usize, y: usize) -> bool {
    let shape_height = shape.len();
    let shape_width = shape[0].len();

    for i in 0..shape_height {
        for j in 0..shape_width {
            // If this cell of the shape is filled and grid cell is already occupied
            if shape[i][j] && grid[x + i][y + j] {
                return false;
            }
        }
    }
    true
}

/// Places a shape on the grid at position (x, y)
fn place(grid: &mut [Vec<bool>], shape: &[Vec<bool>], x: usize, y: usize) {
    let shape_height = shape.len();
    let shape_width = shape[0].len();

    for i in 0..shape_height {
        for j in 0..shape_width {
            if shape[i][j] {
                grid[x + i][y + j] = true;
            }
        }
    }
}

/// Removes a shape from the grid at position (x, y)
fn unplace(grid: &mut [Vec<bool>], shape: &[Vec<bool>], x: usize, y: usize) {
    let shape_height = shape.len();
    let shape_width = shape[0].len();

    for i in 0..shape_height {
        for j in 0..shape_width {
            if shape[i][j] {
                grid[x + i][y + j] = false;
            }
        }
    }
}

/// Solves part 1: Count how many regions can fit all their shapes
#[allow(unused)]
pub(crate) fn resolve_part1(lines: &[String]) -> Result<usize, Box<dyn Error>> {
    let shapes = parse_shapes(lines);

    // Find where region definitions start (lines containing 'x' dimensions)
    let region_start = lines
        .iter()
        .position(|line| line.contains('x'))
        .unwrap_or(0);

    let region_lines = &lines[region_start..];

    // Test each region to see if all shapes fit
    let count = region_lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .filter(|line| {
            let (width, height, counts) = parse_region(line);
            can_fit(&shapes, width, height, counts)
        })
        .count();

    Ok(count)
}

/// Solves part 2: Not yet implemented
#[allow(unused)]
pub(crate) fn resolve_part2(_lines: &[String]) -> Result<usize, Box<dyn Error>> {
    Ok(0)
}

pub(crate) fn run() {
    match read_input() {
        Ok(inputs) => {
            match resolve_part1(&inputs) {
                Ok(result) => print!("Day 12 Part 1: {}, ", result),
                Err(e) => println!("day12 part1 error: {}", e),
            }
            match resolve_part2(&inputs) {
                Ok(result) => println!("Day 12 Part 2: {}", result),
                Err(e) => println!("day12 part2 error: {}", e),
            }
        }
        Err(e) => println!("day12 read input error: {}", e),
    }
}