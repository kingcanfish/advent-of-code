use std::error::Error;
use std::path::PathBuf;

#[allow(unused)]
pub(crate) fn read_input() -> Result<Vec<(i64, i64)>, Box<dyn Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("inputs")
        .join("day2.txt");
    let content = std::fs::read_to_string(path)?;
    let line = content.lines().next().ok_or("No lines in input file")?;
    let result: Result<Vec<(i64, i64)>, _> = line
        .split(',')
        .map(|pair| {
            let parts: Vec<&str> = pair.split('-').collect();
            if parts.len() != 2 {
                return Err(format!("Invalid pair format: {}", pair).into());
            }
            let a: i64 = parts[0].parse()?;
            let b: i64 = parts[1].parse()?;
            Ok((a, b))
        })
        .collect();
    result
}

#[allow(unused)]
pub(crate) fn solve_part1(input: &[(i64, i64)]) -> i64 {
    let mut result = 0i64;
    input.iter().for_each(|(a, b)| {
        for i in *a..(*b + 1) {
            let s = i.to_string();
            if s.len() < 2 || s.len() % 2 != 0 {
                continue;
            }
            let mid = s.len() / 2;

            if (s[0..mid] == s[mid..]) {
                result += i;
            }
        }
    });
    result
}

#[allow(unused)]
pub(crate) fn solve_part2(input: &[(i64, i64)]) -> i64 {
    let mut result = 0i64;
    input.iter().for_each(|(a, b)| {
        for i in *a..(*b + 1) {
            if is_repeated_elegant_math(i) {
                result += i;
            }
        }
    });
    result
}

fn is_repeated_elegant_math(n: i64) -> bool {
    let s = n.to_string();
    let len = s.len();

    if len < 2 {
        return false;
    }

    // 关键洞察：如果字符串由重复模式组成，
    // 那么 (s + s)[1..2*len-1] 应该包含原始字符串 s
    let doubled = format!("{}{}", s, s);
    doubled[1..doubled.len() - 1].contains(&s)
}

pub(crate) fn run() {
    let inputs = read_input().unwrap();
    println!(
        "day2 part1: {:?}, part2: {:?}",
        solve_part1(&inputs),
        solve_part2(&inputs)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        let input = read_input().unwrap();
        let result = solve_part1(&input);
        println!("Part 1: {}", result);
    }
    #[test]
    fn test_example1() {
        let input = vec![(11, 22), (95, 115), (998, 1012)];
        let result = solve_part1(&input);
        assert_eq!(result, 2);
    }
    #[test]
    fn test_example2() {
        let input = vec![(998, 1012)];
        let result = solve_part1(&input);
        assert_eq!(result, 0);
    }
    #[test]
    fn test_part2() {
        let input = read_input().unwrap();
        let result = solve_part2(&input);
        println!("Part 2: {}", result);
    }
}
