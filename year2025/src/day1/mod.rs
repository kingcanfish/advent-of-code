use std::path::PathBuf;

#[allow(unused)]
pub(crate) fn read_input() -> Vec<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("inputs")
        .join("day1.txt");
    std::fs::read_to_string(path)
        .expect("Failed to read input.txt")
        .lines()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
}

use std::error::Error;

/// Resolves the Day 1 puzzle by processing the input lines.
///
/// Each non-empty line starts with 'L' or another character, followed by a number.
/// The sum is updated based on the direction and delta, modulo 100.
/// Counts how many times the sum reaches zero.
///
/// # Arguments
///
/// * `inputs` - A slice of strings representing the input lines.
///
/// # Returns
///
/// * `Result<i64, Box<dyn Error>>` - The count of times sum reached zero, or an error.
#[allow(unused)]
pub(crate) fn resolve_part1(inputs: &[String]) -> Result<i64, Box<dyn Error>> {
    let (count, _) = inputs.iter().filter(|line| !line.is_empty()).try_fold(
        (0i64, 50i64),
        |(count, sum), line| {
            let first_char = line.chars().next().ok_or("Line is empty")?;
            let delta_str = &line[1..];
            let delta = delta_str.parse::<i64>()? % 100;
            let new_sum = if first_char == 'L' {
                (100 + sum - delta) % 100
            } else {
                (100 + sum + delta) % 100
            };
            let new_count = if new_sum == 0 { count + 1 } else { count };
            Ok::<_, Box<dyn Error>>((new_count, new_sum))
        },
    )?;
    Ok(count)
}

#[allow(unused)]
pub(crate) fn resolve_part2(inputs: &[String]) -> Result<i64, Box<dyn Error>> {
    let mut current_position = 50;
    let mut zero_count = 0;

    for line in inputs {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let (direction, amount) = parse_instruction(line)?;

        match direction {
            'R' => {
                // Left rotation (counterclockwise, increasing values)
                // Count how many times we pass through 0
                let start = current_position;
                let end = (current_position + amount) % 100;

                // Calculate how many complete cycles (0-99) we make
                let complete_cycles: i64 = amount / 100;
                zero_count += complete_cycles;

                // Check if we pass through 0 in the partial rotation
                let remaining = amount % 100;
                if remaining > 0 {
                    // We pass through 0 if start + remaining >= 100
                    // which means we wrap around
                    if start + remaining >= 100 {
                        zero_count += 1;
                    }
                }

                current_position = end;
            }
            'L' => {
                // Right rotation (clockwise, decreasing values)
                // Count how many times we pass through 0
                let start = current_position;

                // Calculate how many complete cycles we make
                let complete_cycles = amount / 100;
                zero_count += complete_cycles;

                // Check if we pass through 0 in the partial rotation
                let remaining = amount % 100;
                if remaining > 0 {
                    // We pass through 0 if we need to go past it
                    // Going right from start by remaining steps
                    if remaining >= start && start > 0 {
                        zero_count += 1;
                    }
                }

                current_position = (current_position + 100 - (amount % 100)) % 100;
            }
            _ => return Err("Invalid direction".into()),
        }
    }

    Ok(zero_count)
}

pub(crate) fn run() {
    let inputs = read_input();
    println!(
        "day1 part1: {:?}, part2: {:?}",
        resolve_part1(&inputs),
        resolve_part2(&inputs)
    );
}
fn parse_instruction(line: &str) -> Result<(char, i64), Box<dyn Error>> {
    let direction = line.chars().next().ok_or("Invalid input")?;
    let amount = line[1..].parse()?;
    Ok((direction, amount))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_day1() {
        let inputs = read_input();
        let result = resolve_part1(&inputs).unwrap();
        println!("part1: {result}");
        assert_eq!(result, 1182);

        let result = resolve_part2(&inputs).unwrap();
        println!("part2: {result}");
        assert_eq!(result, 6907);
    }

    #[test]
    fn test_example() {
        let inputs = vec![
            "L68".to_string(),
            "L30".to_string(),
            "R48".to_string(),
            "L5".to_string(),
            "R60".to_string(),
            "L55".to_string(),
            "L1".to_string(),
            "L99".to_string(),
            "R14".to_string(),
            "L82".to_string(),
        ];

        assert_eq!(resolve_part2(&inputs).unwrap(), 6);
    }

    #[test]
    fn test_simple_cases() {
        // From 50, L50 reaches 0 (passes through it)
        let inputs = vec!["L50".to_string()];
        assert_eq!(resolve_part2(&inputs).unwrap(), 1);

        // From 50, R50 reaches 0 (passes through it)
        let inputs = vec!["R50".to_string()];
        assert_eq!(resolve_part2(&inputs).unwrap(), 1);

        // From 50, L30 reaches 80 (doesn't pass through 0)
        let inputs = vec!["L30".to_string()];
        assert_eq!(resolve_part2(&inputs).unwrap(), 0);

        // From 50, R30 reaches 20 (doesn't pass through 0)
        let inputs = vec!["R30".to_string()];
        assert_eq!(resolve_part2(&inputs).unwrap(), 0);
    }
}
