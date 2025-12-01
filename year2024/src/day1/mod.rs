mod part1;
mod part2;

use std::fs::File;
use std::io::{BufRead, BufReader, Result};
#[allow(unused)]
pub(crate) fn read_input(file_name: &str) -> Result<(Vec<i64>, Vec<i64>)> {
    let file = File::open(file_name)?;
    let reader = BufReader::new(file);

    reader
        .lines()
        .try_fold((Vec::new(), Vec::new()), |(mut left, mut right), line| {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() >= 2 {
                let left_num: i64 = parts[0]
                    .parse()
                    .map_err(|_| invalid_data("Invalid number in left column"))?;
                let right_num: i64 = parts[1]
                    .parse()
                    .map_err(|_| invalid_data("Invalid number in right column"))?;

                left.push(left_num);
                right.push(right_num);
            }

            Ok((left, right))
        })
}

#[allow(unused)]
fn invalid_data(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_part1() {
        let (left, right) = read_input("inputs/day1.txt").unwrap();
        let result = part1::resolve(&left, &right);
        println!("{}", result);
    }

    #[test]
    fn test_part2() {
        let (left, right) = read_input("inputs/day1.txt").unwrap();
        let result = part2::resolve(&left, &right);
        println!("{}", result);
    }
}
