mod part1;
mod part2;

use std::fs::File;
use std::io::{BufRead, BufReader, Error};

#[allow(unused)]
pub(crate) fn read_input(file_name: &str) -> Result<Vec<Vec<i64>>, Error> {
    let file = File::open(file_name)?;
    let reader = BufReader::new(file);
    let mut result = vec![];
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let elements = line
            .split_whitespace()
            .map(|s| s.parse::<i64>().unwrap())
            .collect::<Vec<_>>();
        result.push(elements);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_day2_part1() {
        let lines = read_input("inputs/day2.txt").unwrap();
        let result = part1::resolve(&lines);
        println!("Part 1: {}", result);
    }

    #[test]
    fn test_day2_part2() {
        let lines = read_input("inputs/day2.txt").unwrap();
        let result = part2::resolve(&lines);
        println!("Part 2: {}", result);
    }
}
