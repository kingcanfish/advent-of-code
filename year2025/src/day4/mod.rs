use crate::file::read_file;
use std::error::Error;

pub(crate) fn unmarshal_input(content: &String) -> Result<Vec<Vec<char>>, Box<dyn Error>> {
    let result: Vec<Vec<char>> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.chars().collect())
        .collect();

    Ok(result)
}

// 提取公共函数用于计算邻居数量
fn count_neighbors(grid: &[Vec<char>], row: usize, col: usize) -> i32 {
    let mut count = 0;

    // 检查周围的8个位置
    for dx in -1..=1 {
        for dy in -1..=1 {
            if dx == 0 && dy == 0 {
                continue; // 跳过当前位置
            }

            let ni = row as i32 + dx;
            let nj = col as i32 + dy;

            // 使用 check 函数检查位置是否有效且字符为 '@'
            if check(grid, (ni as usize, nj as usize)) {
                count += 1;
            }
        }
    }

    count
}

pub(crate) fn resolve_part1(input: &String) -> i64 {
    let input = unmarshal_input(input).unwrap();
    let mut cnt = 0i64;

    for (i, row) in input.iter().enumerate() {
        for (j, &cell) in row.iter().enumerate() {
            if cell != '@' {
                continue;
            }

            // 如果邻居中 '@' 字符数量小于4，则计数器加1
            if count_neighbors(&input, i, j) < 4 {
                cnt += 1;
            }
        }
    }

    cnt
}

pub fn resolve_part2(input: &String) -> i64 {
    let mut input = unmarshal_input(input).unwrap();

    let mut total_removed = 0i64;

    loop {
        let mut positions_to_remove = Vec::new();
        // 找出这一轮所有可以移除的@
        for (i, row) in input.iter().enumerate() {
            for (j, &cell) in row.iter().enumerate() {
                if cell != '@' {
                    continue;
                }

                // 如果邻居中 '@' 字符数量小于4，则记录这个位置准备移除
                if count_neighbors(&input, i, j) < 4 {
                    positions_to_remove.push((i, j));
                }
            }
        }

        // 如果这一轮没有可以移除的@，则结束循环
        if positions_to_remove.is_empty() {
            break;
        }
        total_removed += positions_to_remove.len() as i64;

        // 移除所有标记的位置
        for (i, j) in positions_to_remove {
            input[i][j] = '.';
        }
    }

    total_removed
}

fn check(input: &[Vec<char>], position: (usize, usize)) -> bool {
    let (x, y) = position;
    if x >= input.len() || y >= input[x].len() {
        return false;
    }
    input[x][y] == '@'
}

pub(crate) fn run() {
    let input = read_file("day4.txt").unwrap();
    println!(
        "day 4 Part 1: {}, Part2: {}",
        resolve_part1(&input),
        resolve_part2(&input)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        let grid = r#"
        ..@@.@@@@.
        @@@.@.@.@@
        @@@@@.@.@@
        @.@@@@..@.
        @@.@@@@.@@
        .@@@@@@@.@
        .@.@.@.@@@
        @.@@@.@@@@
        .@@@@@@@@.
        @.@.@@@.@.
        "#
        .to_string();
        assert_eq!(resolve_part1(&grid), 13);
    }
}
