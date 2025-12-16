use crate::file::read_file;
use std::error::Error;

#[derive(Debug)]
struct Problem {
    numbers: Vec<i64>,
    operation: Operation,
}

#[derive(Debug)]
enum Operation {
    Add,
    Multiply,
}

impl Problem {
    fn solve(&self) -> i64 {
        match self.operation {
            Operation::Add => self.numbers.iter().sum(),
            Operation::Multiply => self.numbers.iter().product(),
        }
    }
}

fn unmarshal_input(content: &str) -> Result<Vec<Problem>, Box<dyn Error>> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(vec![]);
    }

    // Split each line by whitespace to get columns
    let rows: Vec<Vec<&str>> = lines
        .iter()
        .map(|line| line.split_whitespace().collect())
        .collect();

    if rows.is_empty() {
        return Ok(vec![]);
    }

    let num_problems = rows[0].len();
    let mut problems = Vec::new();

    // Process each column (problem)
    for col in 0..num_problems {
        let mut numbers = Vec::new();
        let mut operation = None;

        for row in &rows {
            if col >= row.len() {
                continue;
            }

            let item = row[col];
            if item == "+" {
                operation = Some(Operation::Add);
            } else if item == "*" {
                operation = Some(Operation::Multiply);
            } else if let Ok(num) = item.parse::<i64>() {
                numbers.push(num);
            }
        }

        if let Some(op) = operation
            && !numbers.is_empty()
        {
            problems.push(Problem {
                numbers,
                operation: op,
            });
        }
    }

    Ok(problems)
}

fn unmarshal_input_part2(content: &str) -> Result<Vec<Problem>, Box<dyn Error>> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(vec![]);
    }

    let max_width = lines.iter().map(|line| line.len()).max().unwrap_or(0);

    // Build character grid
    let mut grid: Vec<Vec<char>> = vec![vec![' '; max_width]; lines.len()];
    for (row, line) in lines.iter().enumerate() {
        for (col, ch) in line.chars().enumerate() {
            grid[row][col] = ch;
        }
    }

    let mut problems = Vec::new();
    let mut col = max_width;

    while col > 0 {
        // Skip separator columns from right
        while col > 0 && grid.iter().all(|row| row[col - 1] == ' ') {
            col -= 1;
        }

        if col == 0 {
            break;
        }

        // Find problem start (left boundary)
        let end = col;
        while col > 0 && grid.iter().any(|row| row[col - 1] != ' ') {
            col -= 1;
        }
        let start = col;

        // Extract operator from last row
        let op_str: String = grid.last().unwrap()[start..end]
            .iter()
            .collect::<String>()
            .trim()
            .to_string();

        let operation = match op_str.as_str() {
            "+" => Some(Operation::Add),
            "*" => Some(Operation::Multiply),
            _ => None,
        };

        if operation.is_none() {
            continue;
        }

        // Read numbers from right to left, each column is one number
        let mut numbers = Vec::new();
        for c in (start..end).rev() {
            let mut digits = String::new();
            for row in grid.iter().take(grid.len() - 1) {
                // Exclude operator row
                let ch = row[c];
                if ch.is_numeric() {
                    digits.push(ch);
                }
            }

            if !digits.is_empty()
                && let Ok(num) = digits.parse::<i64>()
            {
                numbers.push(num);
            }
        }

        if let Some(op) = operation
            && !numbers.is_empty()
        {
            problems.push(Problem {
                numbers,
                operation: op,
            });
        }
    }

    Ok(problems)
}

fn resolve_part1(input: &str) -> i64 {
    let problems = unmarshal_input(input).unwrap();
    problems.iter().map(|p| p.solve()).sum()
}

fn resolve_part2(input: &str) -> i64 {
    let problems = unmarshal_input_part2(input).unwrap();
    problems.iter().map(|p| p.solve()).sum()
}

pub(crate) fn run() {
    let input = read_file("day6.txt").unwrap();
    println!(
        "Day 6 Part 1: {}, Part 2: {}",
        resolve_part1(&input),
        resolve_part2(&input)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        let input = r#"123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +  "#
            .to_string();
        assert_eq!(resolve_part1(&input), 4277556);
    }

    #[test]
    fn test_part2() {
        let input = r#"123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +  "#
            .to_string();
        assert_eq!(resolve_part2(&input), 3263827);
    }
}
