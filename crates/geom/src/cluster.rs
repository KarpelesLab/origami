//! Agglomerative single-linkage clustering of trajectory frames by
//! Cα RMSD. Used by `origami analyze --cluster-cutoff` to group a
//! trajectory's conformations into fold basins.
//!
//! Algorithm:
//!   1. Compute pairwise Cα RMSD between every frame (O(N²) calls
//!      to `rmsd_ca`; for small proteins each call is microseconds,
//!      so a 1 k-frame trajectory runs in ~1 s).
//!   2. Each frame starts as its own cluster.
//!   3. Repeatedly find the closest pair of clusters (min-RMSD
//!      between any frame in one cluster and any frame in the
//!      other). If the min-distance ≤ `cutoff_a`, merge them.
//!      Otherwise stop.
//!   4. Return per-frame cluster labels (0, 1, 2, …) in size-
//!      decreasing order so cluster 0 is the largest basin, 1 is
//!      the second-largest, etc.
//!
//! Single-linkage produces "chains" of conformations connected by
//! pairwise similarity — appropriate for detecting whether a fold
//! trajectory visits one basin or several. The cutoff in Å sets the
//! resolution; 2.0 Å is a reasonable default for distinguishing
//! native from non-native basins on small proteins.

use crate::rmsd::rmsd_ca;
use crate::structure::Structure;

/// Cluster a trajectory's frames by Cα RMSD with `cutoff_a` Å as the
/// single-linkage merge threshold. Returns one cluster label per
/// frame, labels assigned in cluster-size-descending order
/// (largest = 0). Frames with no computable RMSD (different residue
/// counts) get their own singleton clusters.
pub fn cluster_trajectory(frames: &[Structure], cutoff_a: f64) -> Vec<usize> {
    let n = frames.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Pairwise RMSD matrix. Symmetric; we fill the upper triangle.
    let mut d = vec![f64::INFINITY; n * n];
    for i in 0..n {
        d[i * n + i] = 0.0;
        for j in (i + 1)..n {
            let r = rmsd_ca(&frames[i], &frames[j]).unwrap_or(f64::INFINITY);
            d[i * n + j] = r;
            d[j * n + i] = r;
        }
    }

    // Per-frame cluster representative — initially each frame is its
    // own cluster. We track the cluster id in `cluster[i]` and use
    // union-find-style merge.
    let mut cluster: Vec<usize> = (0..n).collect();

    // Iterative single-linkage: at each step find the closest pair
    // (i, j) with cluster[i] ≠ cluster[j] and d[i][j] ≤ cutoff. Merge
    // by relabeling. Stop when no such pair exists.
    loop {
        let mut best = cutoff_a;
        let mut best_pair: Option<(usize, usize)> = None;
        for i in 0..n {
            for j in (i + 1)..n {
                if cluster[i] == cluster[j] {
                    continue;
                }
                if d[i * n + j] < best {
                    best = d[i * n + j];
                    best_pair = Some((i, j));
                }
            }
        }
        let Some((i, j)) = best_pair else {
            break;
        };
        let (old_label, new_label) = (cluster[j].max(cluster[i]), cluster[j].min(cluster[i]));
        for c in cluster.iter_mut() {
            if *c == old_label {
                *c = new_label;
            }
        }
    }

    // Compact the labels so they're 0..k and order them by cluster
    // size (largest cluster gets label 0).
    let mut counts: std::collections::BTreeMap<usize, usize> = Default::default();
    for &c in &cluster {
        *counts.entry(c).or_default() += 1;
    }
    let mut sorted: Vec<(usize, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let remap: std::collections::HashMap<usize, usize> = sorted
        .iter()
        .enumerate()
        .map(|(new, (old, _))| (*old, new))
        .collect();
    cluster.iter().map(|c| *remap.get(c).unwrap_or(&0)).collect()
}

/// Helper: aggregate cluster labels into a `(label → size)` list,
/// sorted by size descending.
pub fn cluster_sizes(labels: &[usize]) -> Vec<(usize, usize)> {
    let mut counts: std::collections::BTreeMap<usize, usize> = Default::default();
    for &c in labels {
        *counts.entry(c).or_default() += 1;
    }
    let mut sorted: Vec<(usize, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_extended_chain;
    use chem::AminoAcid;

    #[test]
    fn identical_frames_one_cluster() {
        let s = build_extended_chain(&[
            AminoAcid::Ala,
            AminoAcid::Gly,
            AminoAcid::Ala,
        ])
        .unwrap();
        let frames = vec![s.clone(), s.clone(), s.clone()];
        let labels = cluster_trajectory(&frames, 0.1);
        assert_eq!(labels, vec![0, 0, 0]);
        assert_eq!(cluster_sizes(&labels), vec![(0, 3)]);
    }

    #[test]
    fn distinct_frames_separate_clusters_when_cutoff_tight() {
        // Two visibly different conformations of the same sequence:
        // one extended, one with the C-terminal Cα shoved 10 Å along
        // x. Even cluster-singletons can't merge through a 1 Å cutoff.
        let s_base = build_extended_chain(&[
            AminoAcid::Ala,
            AminoAcid::Ala,
            AminoAcid::Ala,
            AminoAcid::Ala,
        ])
        .unwrap();
        let mut s_warped = s_base.clone();
        for a in &mut s_warped.residues[3].atoms {
            if a.name == "CA" {
                a.position.x += 10.0;
            }
        }
        let frames = vec![s_base.clone(), s_warped.clone(), s_base.clone()];
        let labels = cluster_trajectory(&frames, 1.0);
        let sizes = cluster_sizes(&labels);
        assert_eq!(sizes.iter().map(|(_, n)| *n).collect::<Vec<_>>(), vec![2, 1]);
    }
}
