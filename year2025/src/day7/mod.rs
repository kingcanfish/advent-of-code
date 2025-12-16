use crate::file::read_file;
use std::collections::{HashMap, HashSet};

fn parse_input(content: &str) -> (Vec<Vec<char>>, usize) {
    let grid: Vec<Vec<char>> = content.lines().map(|line| line.chars().collect()).collect();

    // Find the starting column 'S'
    let mut start_col = 0;
    for line in grid.iter() {
        for (col, &ch) in line.iter().enumerate() {
            if ch == 'S' {
                start_col = col;
                break;
            }
        }
    }

    (grid, start_col)
}

fn count_splits(grid: &[Vec<char>], start_col: usize) -> usize {
    let rows = grid.len();
    let cols = if rows > 0 { grid[0].len() } else { 0 };

    let mut split_count = 0;
    let mut current_cols = HashSet::new();
    current_cols.insert(start_col);

    // 逐行处理
    for row in 0..rows - 1 {
        let mut next_cols = HashSet::new();

        for &col in &current_cols {
            let next_row = row + 1;
            let next_cell = grid[next_row][col];

            if next_cell == '.' || next_cell == 'S' {
                // 继续向下，同列
                next_cols.insert(col);
            } else if next_cell == '^' {
                // 遇到分裂器，计数
                split_count += 1;

                // 分裂到左右两列
                if col > 0 {
                    next_cols.insert(col - 1);
                }
                if col + 1 < cols {
                    next_cols.insert(col + 1);
                }
            }
        }

        current_cols = next_cols;

        // 如果没有 beam 了，提前结束
        if current_cols.is_empty() {
            break;
        }
    }

    split_count
}

fn count_timelines(grid: &[Vec<char>], start_col: usize) -> usize {
    let rows = grid.len();
    let cols = if rows > 0 { grid[0].len() } else { 0 };

    // 使用 HashMap 而不是二维数组，节省空间
    let mut current_paths: HashMap<usize, usize> = HashMap::new();
    current_paths.insert(start_col, 1);

    for row in 0..rows - 1 {
        let mut next_paths: HashMap<usize, usize> = HashMap::new();

        for (&col, &path_count) in &current_paths {
            let next_row = row + 1;
            let next_cell = grid[next_row][col];

            if next_cell == '.' || next_cell == 'S' {
                // 继续向下，路径数传递
                *next_paths.entry(col).or_insert(0) += path_count;
            } else if next_cell == '^' {
                // 分裂器：时间线分裂成两条
                // 向左分裂
                if col > 0 {
                    *next_paths.entry(col - 1).or_insert(0) += path_count;
                }
                // 向右分裂
                if col + 1 < cols {
                    *next_paths.entry(col + 1).or_insert(0) += path_count;
                }
            }
        }

        current_paths = next_paths;
        if current_paths.is_empty() {
            break;
        }
    }

    // 返回所有路径数总和
    current_paths.values().sum()
}

fn resolve_part1(input: &str) -> usize {
    let (grid, start_col) = parse_input(input);
    count_splits(&grid, start_col)
}

fn resolve_part2(input: &str) -> usize {
    let (grid, start_col) = parse_input(input);
    count_timelines(&grid, start_col)
}

pub(crate) fn run() {
    let input = read_file("day7.txt").unwrap();
    println!(
        "Day 7 Part 1: {}, Part 2: {}",
        resolve_part1(&input),
        resolve_part2(&input)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        let input = r#".......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
..............."#;
        assert_eq!(resolve_part1(input), 21);
    }

    #[test]
    fn test_part2() {
        let input = r#".......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
..............."#;
        assert_eq!(resolve_part2(input), 40);
    }
}
