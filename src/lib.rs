use pyo3::prelude::*;
use pyo3::types::PyBytes;
use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use osmpbf::{ElementReader, Element};
use rustc_hash::FxHashMap; // Fast HashMap
use std::sync::Arc;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct LinkData {
    start_loc: [f64; 2],
    end_loc: [f64; 2],
    start_node: String,
    end_node: String,
    roadtype: String,
    length: Option<String>,
    meshcode: Option<String>,
}

#[pyclass(module = "spatial_lookup")]
// Removed #[derive(Serialize, Deserialize)] here to handle it manually in __getstate__
struct SpatialIndex {
    tree: KdTree<f64, usize, [f64; 2]>,
    links: Vec<LinkData>,
}

#[pymethods]
impl SpatialIndex {
    #[new]
    fn new() -> Self {
        SpatialIndex {
            tree: KdTree::new(2),
            links: Vec::new(),
        }
    }

    /// Build a SpatialIndex directly from Python arrays — no PBF file needed.
    ///
    /// Used by the parity benchmark so both engines can be tested on the same
    /// synthetic dataset without authoring an OSM PBF.
    ///
    /// Parameters (all must be the same length N):
    ///   start_lons, start_lats  — start coordinates of each link
    ///   end_lons,   end_lats    — end   coordinates of each link
    ///   start_nodes, end_nodes  — node-ID strings
    ///   roadtypes               — OSM highway tag strings
    ///   lengths                 — optional length strings (empty str → None)
    ///   meshcodes               — optional meshcode strings (empty str → None)
    #[staticmethod]
    #[pyo3(signature = (start_lons, start_lats, end_lons, end_lats,
                        start_nodes, end_nodes, roadtypes,
                        lengths, meshcodes))]
    fn from_arrays(
        start_lons: Vec<f64>,
        start_lats: Vec<f64>,
        end_lons: Vec<f64>,
        end_lats: Vec<f64>,
        start_nodes: Vec<String>,
        end_nodes: Vec<String>,
        roadtypes: Vec<String>,
        lengths: Vec<String>,
        meshcodes: Vec<String>,
    ) -> PyResult<Self> {
        let n = start_lons.len();
        if start_lats.len() != n || end_lons.len() != n || end_lats.len() != n
            || start_nodes.len() != n || end_nodes.len() != n
            || roadtypes.len() != n || lengths.len() != n || meshcodes.len() != n
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All input arrays must have the same length",
            ));
        }

        let mut index = SpatialIndex {
            tree: KdTree::new(2),
            links: Vec::with_capacity(n),
        };

        for i in 0..n {
            let link = LinkData {
                start_loc: [start_lons[i], start_lats[i]],
                end_loc:   [end_lons[i],   end_lats[i]],
                start_node: start_nodes[i].clone(),
                end_node:   end_nodes[i].clone(),
                roadtype:   roadtypes[i].clone(),
                length:   if lengths[i].is_empty()   { None } else { Some(lengths[i].clone())   },
                meshcode: if meshcodes[i].is_empty() { None } else { Some(meshcodes[i].clone()) },
            };
            let start = link.start_loc;
            index.links.push(link);
            let _ = index.tree.add(start, i);
        }

        Ok(index)
    }

    #[pyo3(signature = (path))]
    fn load_osm_pbf(&mut self, path: String) -> PyResult<usize> {
        // --- PASS 1: NODES (PARALLEL) ---
        let reader = ElementReader::from_path(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to open PBF: {}", e))
        })?;

        let node_map = reader.par_map_reduce(
            |element| {
                let mut local_map = FxHashMap::default();
                match element {
                    Element::Node(node) => {
                        local_map.insert(node.id(), (node.lon(), node.lat()));
                    },
                    Element::DenseNode(node) => {
                        local_map.insert(node.id(), (node.lon(), node.lat()));
                    },
                    _ => {}
                }
                local_map
            },
            || FxHashMap::default(),
            |mut a, b| {
                a.extend(b);
                a
            }
        ).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("PBF Node Error: {}", e)))?;

        let node_map_arc = Arc::new(node_map);

        // --- PASS 2: WAYS (PARALLEL) ---
        let reader_pass2 = ElementReader::from_path(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to reopen PBF: {}", e))
        })?;

        let new_links = reader_pass2.par_map_reduce(
            |element| {
                let mut local_links = Vec::new();
                if let Element::Way(way) = element {
                    let mut highway = "unknown".to_string();
                    let mut length = None;
                    let mut meshcode = None;

                    for (k, v) in way.tags() {
                        match k {
                            "highway"  => highway  = v.to_string(),
                            "length"   => length   = Some(v.to_string()),
                            "ref:mesh" => meshcode = Some(v.to_string()),
                            _ => {}
                        }
                    }

                    // Derive start/end node IDs directly from the way's node refs —
                    // the standard OSM encoding; no vendor tag required.
                    let refs: Vec<i64> = way.refs().collect();
                    if let (Some(&first_id), Some(&last_id)) = (refs.first(), refs.last()) {
                        if let (Some(&start_coord), Some(&end_coord)) = (node_map_arc.get(&first_id), node_map_arc.get(&last_id)) {
                            local_links.push(LinkData {
                                start_loc: [start_coord.0, start_coord.1],
                                end_loc:   [end_coord.0,   end_coord.1],
                                start_node: first_id.to_string(),
                                end_node:   last_id.to_string(),
                                roadtype:   highway,
                                length:     length,
                                meshcode:   meshcode,
                            });
                        }
                    }
                }
                local_links
            },
            || Vec::new(),
            |mut a, b| {
                a.extend(b);
                a
            }
        ).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("PBF Way Error: {}", e)))?;

        // --- PASS 3: BUILD TREE (Main Thread) ---
        let count = new_links.len();
        self.links.reserve(count);
        
        for link in new_links.into_iter() {
            let start = link.start_loc;
            let idx = self.links.len(); 
            self.links.push(link);
            let _ = self.tree.add(start, idx);
        }

        Ok(count)
    }

    #[pyo3(signature = (wkt_list, radius_deg, d_limit_meters))]
    fn find_match_bulk_rayon(
        &self,
        py: Python<'_>,
        wkt_list: Vec<Option<String>>, 
        radius_deg: f64,
        d_limit_meters: f64
    ) -> (Vec<Option<String>>, Vec<Option<String>>, Vec<Option<String>>, Vec<Option<String>>, Vec<Option<String>>) {
        
        py.allow_threads(|| {
            let results: Vec<Option<(String, String, String, Option<String>, Option<String>)>> = wkt_list
                .par_iter()
                .map(|wkt_opt| {
                    let wkt = wkt_opt.as_ref()?;
                    let nums: Vec<f64> = wkt
                        .split(|c: char| !c.is_numeric() && c != '.' && c != '-')
                        .filter(|s| !s.is_empty())
                        .filter_map(|s| s.parse::<f64>().ok())
                        .collect();

                    if nums.len() < 4 { return None; }
                    self.find_match_internal(nums[0], nums[1], nums[2], nums[3], radius_deg, d_limit_meters)
                })
                .collect();

            let mut start_nodes = Vec::with_capacity(results.len());
            let mut end_nodes = Vec::with_capacity(results.len());
            let mut roadtypes = Vec::with_capacity(results.len());
            let mut lengths = Vec::with_capacity(results.len());
            let mut meshcodes = Vec::with_capacity(results.len());

            for res in results {
                match res {
                    Some((s, e, r, l, m)) => {
                        start_nodes.push(Some(s));
                        end_nodes.push(Some(e));
                        roadtypes.push(Some(r));
                        lengths.push(l);
                        meshcodes.push(m);
                    },
                    None => {
                        start_nodes.push(None);
                        end_nodes.push(None);
                        roadtypes.push(None);
                        lengths.push(None);
                        meshcodes.push(None);
                    }
                }
            }
            (start_nodes, end_nodes, roadtypes, lengths, meshcodes)
        })
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = bincode::serialize(&self.links).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization failed: {}", e))
        })?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let decoded_links: Vec<LinkData> = bincode::deserialize(state.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Deserialization failed: {}", e))
        })?;
        
        self.links = decoded_links;
        
        self.tree = KdTree::new(2);
        for (idx, link) in self.links.iter().enumerate() {
            let _ = self.tree.add(link.start_loc, idx);
        }
        
        Ok(())
    }
}

impl SpatialIndex {
    fn find_match_internal(
        &self,
        start_lon: f64, start_lat: f64,
        end_lon: f64, end_lat: f64,
        radius_deg: f64,
        d_limit_meters: f64
    ) -> Option<(String, String, String, Option<String>, Option<String>)> {
        
        let p_start = [start_lon, start_lat];
        let p_end = [end_lon, end_lat];
        let radius_sq = radius_deg * radius_deg;

        let idxs_start = self.tree.within(&p_start, radius_sq, &squared_euclidean).unwrap_or_default();
        let idxs_end = self.tree.within(&p_end, radius_sq, &squared_euclidean).unwrap_or_default();

        let mut best_candidate: Option<(f64, &LinkData)> = None;
        let mut seen = std::collections::HashSet::new();

        for (_, &idx) in idxs_start.iter().chain(idxs_end.iter()) {
            if seen.contains(&idx) { continue; }
            seen.insert(idx);

            let link = &self.links[idx];
            
            let d_norm_start = haversine(start_lon, start_lat, link.start_loc[0], link.start_loc[1]);
            let d_norm_end = haversine(end_lon, end_lat, link.end_loc[0], link.end_loc[1]);
            let d_rev_start = haversine(start_lon, start_lat, link.end_loc[0], link.end_loc[1]);
            let d_rev_end = haversine(end_lon, end_lat, link.start_loc[0], link.start_loc[1]);

            let mut current_dist = None;

            if d_norm_start <= d_limit_meters && d_norm_end <= d_limit_meters {
                current_dist = Some(d_norm_start + d_norm_end);
            } else if d_rev_start <= d_limit_meters && d_rev_end <= d_limit_meters {
                current_dist = Some(d_rev_start + d_rev_end);
            }

            if let Some(dist) = current_dist {
                match best_candidate {
                    None => best_candidate = Some((dist, link)),
                    Some((best_dist, _)) => {
                        if dist < best_dist {
                            best_candidate = Some((dist, link));
                        }
                    }
                }
            }
        }

        best_candidate.map(|(_, link)| {
            (
                link.start_node.clone(), 
                link.end_node.clone(), 
                link.roadtype.clone(), 
                link.length.clone(), 
                link.meshcode.clone()
            )
        })
    }
}

fn haversine(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let r = 6371000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2) + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    r * c
}

#[pymodule]
fn spatial_lookup(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SpatialIndex>()?;
    Ok(())
}
