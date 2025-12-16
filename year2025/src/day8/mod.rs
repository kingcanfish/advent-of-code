use crate::file::read_file;
use std::error::Error;

fn parse_input(content: &str) -> Vec<Point> {
    content
        .lines()
        .map(|line| {
            let parts: Vec<i32> = line.split(',').map(|s| s.trim().parse().unwrap()).collect();
            Point::new(parts[0], parts[1], parts[2])
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Point {
    x: i32,
    y: i32,
    z: i32,
}

impl Point {
    fn new(x: i32, y: i32, z: i32) -> Self {
        Point { x, y, z }
    }

    fn distance(&self, other: &Point) -> f64 {
        let dx = (self.x - other.x) as f64;
        let dy = (self.y - other.y) as f64;
        let dz = (self.z - other.z) as f64;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let mut parent = vec![0; n];
        let size = vec![1; n];
        for (i, p) in parent.iter_mut().enumerate().take(n) {
            *p = i;
        }
        UnionFind {
            parent,
            rank: vec![0; n],
            size,
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let px = self.find(x);
        let py = self.find(y);
        if px == py {
            return;
        }
        if self.rank[px] < self.rank[py] {
            self.parent[px] = py;
            self.size[py] += self.size[px];
        } else {
            self.parent[py] = px;
            self.size[px] += self.size[py];
            if self.rank[px] == self.rank[py] {
                self.rank[px] += 1;
            }
        }
    }

    fn get_sizes(&mut self) -> Vec<usize> {
        let mut sizes = Vec::new();
        for i in 0..self.parent.len() {
            if self.find(i) == i {
                sizes.push(self.size[i]);
            }
        }
        sizes
    }
}

fn solve(points: &[Point], connections: Option<usize>) -> Result<i64, Box<dyn Error>> {
    let n = points.len();
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..n {
        for j in i + 1..n {
            let dist = points[i].distance(&points[j]);
            edges.push((i, j, dist));
        }
    }

    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    let mut uf = UnionFind::new(n);
    let mut last_edge = None;

    if let Some(conn) = connections {
        // part1: connect fixed number
        for (i, j, _) in edges.into_iter().take(conn) {
            uf.union(i, j);
        }

        let mut sizes = uf.get_sizes();
        sizes.sort_by(|a, b| b.cmp(a));

        if sizes.len() < 3 {
            return Err("Not enough circuits".into());
        }

        let product = sizes[0] as i64 * sizes[1] as i64 * sizes[2] as i64;
        Ok(product)
    } else {
        // part2: connect until fully connected
        for (i, j, _) in edges.into_iter() {
            uf.union(i, j);
            last_edge = Some((i, j));
            if uf.get_sizes().len() == 1 {
                break;
            }
        }

        if let Some((i, j)) = last_edge {
            let x1 = points[i].x as i64;
            let x2 = points[j].x as i64;
            Ok(x1 * x2)
        } else {
            Err("No edges".into())
        }
    }
}

fn resolve_part1(input: &str) -> Result<i64, Box<dyn Error>> {
    let points = parse_input(input);
    solve(&points, Some(1000))
}

fn resolve_part2(input: &str) -> Result<i64, Box<dyn Error>> {
    let points = parse_input(input);
    solve(&points, None)
}

pub(crate) fn run() {
    let input = read_file("day8.txt").unwrap();
    match resolve_part1(&input) {
        Ok(result) => print!("Day8 part1: {},", result),
        Err(e) => println!("day8 part1 error: {}", e),
    }
    match resolve_part2(&input) {
        Ok(result) => print!(" part2: {}", result),
        Err(e) => println!("day8 part2 error: {}", e),
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example() {
        let input = r#"162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689"#;

        let points = parse_input(input);
        let result = solve(&points, Some(10)).unwrap();
        assert_eq!(result, 40);
    }

    #[test]
    fn test_example_part2() {
        let input = r#"162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689"#;

        let points = parse_input(input);
        let result = solve(&points, None).unwrap();
        // 最后一个连接的边是 216 和 117，X 坐标乘积 216 * 117 = 25272
        assert_eq!(result, 25272);
    }
}
