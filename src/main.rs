pub mod file_skip_list;
pub mod id_book;
pub mod index_builder;
pub mod inverted_index;
pub mod lazy_merger;
pub mod postings;
pub mod query;
pub mod single_posting;
pub mod tokenizer;
use actix_cors::Cors;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

fn main() {
    // ! BUILD INDEX
    // let doc_id = index_builder::main();
    // ! MERGE BATCHES
    // ! The following code snippet merges the batches of inverted indexes into a multiple sorted inverted index.
    // lazy_merger::main(doc_id);

    // ! Comes searching and ranking now

    println!("Welcome to the Search Engine!");
    let mut search_engine = query::SearchEngine::new();
    loop {
        search_engine.get_query();
        search_engine.search();
    }
    // TODO: Implement search and ranking logic using the query
}

// #[derive(Deserialize)]
// struct SearchRequest {
//     query: String,
// }

// #[derive(Serialize)]
// struct SearchResponse {
//     results: Vec<String>,
//     time: u128,
// }

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     println!("Welcome to the Search Engine!");

//     let search_engine = Arc::new(Mutex::new(query::SearchEngine::new()));

//     HttpServer::new(move || {
//         // Configure CORS middleware
//         let cors = Cors::default()
//             .allowed_origin("http://localhost:4321")
//             .allow_any_method()
//             .allow_any_header()
//             .max_age(3600);

//         App::new()
//             .wrap(cors)
//             .app_data(web::Data::new(search_engine.clone()))
//             .route("/", web::get().to(index))
//             .route("/search", web::post().to(handle_search))
//     })
//     .bind(("127.0.0.1", 3000))?
//     .run()
//     .await
// }

// async fn index() -> impl Responder {
//     HttpResponse::Ok().body("Search Engine API")
// }

// async fn handle_search(
//     payload: web::Json<SearchRequest>,
//     search_engine: web::Data<Arc<Mutex<query::SearchEngine>>>,
// ) -> impl Responder {
//     let mut engine = search_engine.lock().unwrap();

//     // Set the query and perform search
//     engine.set_query(payload.query.clone());
//     let (results, time) = engine.search();

//     // Limit to 5 results
//     let limited_results = results.into_iter().take(5).collect();

//     HttpResponse::Ok().json(SearchResponse {
//         results: limited_results,
//         time: time,
//     })
// }

// ? archive problems :=> Meta tags with nothing, simple txt file with minimal contents
// ? cbcl :=> Numpy arrays, so if the path has an extension which is not really a file then maybe best to ignore it
// ? "https://cbcl.ics.uci.edu/public_data/shilab/ChIPseq_elmira_08272016/peaks/R301-L2-P6-ATGTCA-Sequences_trimmed-mapped_peaks.narrowPeak" gives us data as well
// ? https://cbcl.ics.uci.edu/public_data/shilab/ChIPseq_elmira_08272016/homer-motifs/P2-peaks-motifs/homerMotifs.motifs5

// How are we linking to the dataset files though, usually i would at least link it via the word here and then the link to the file

// ? chenli wordpress xml is fine
// ? computable plant tsv which is data not fine
// ? 2c6 for above ^  has deformed html but it should be valid still somehow
// ? flamingo has a .cc file which is cpp file
// ? grape has a format=txt file which is fine, but hard to detect
// ? mailman has a file which is deformed html but should be valid
// ? sprout has /readme but no md and deformed html with meta tag and nothing in it
// ? ugradforms has deformed
// ?w3 ^
// ? www_cs_uci_edu has a defrmed calendar
// ? www_ics_uci_edu has a .prefs file for preferences, .lif for something weird
// ? ^ continued: .TXT, .R, https://ics.uci.edu/~eppstein/projects/pairs/Data/-i12/m500 ???
// ?above x 2
//  ? Monodego has one file which is super big but with low textual content so, we can do a thing to index all files which are above 5mb 5_000_000 bytes
// ? the path in the file can only end with txt, html, htm, md, xml, xhtml, xhtm, xht, xhtml, xht, xhtm, like when converted to lower case especially or there should be no file extension at all
// ? we need a duplicate checker as well as well a way to store all the lines domains and their paths in an index!
// html + strip must have something to process
// also some undetected files are 85 mb, so what do we do with that

// if !file_contents.contains("HTML")
// && !file_contents.contains("html")
// && !file_contents.contains(".txt")
// && !file_contents.contains("htm")
// && !file_contents.contains(".md")
// && !file_contents.contains("xml")
// && !file_contents.contains("XML")
