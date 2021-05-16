use std::{cmp::max, fs};

use image::{png::PngDecoder, DynamicImage, GenericImageView};
use text_recovery::{
    drop_empty_rectangle_matches, find_geometric_matches_for_single_results,
    find_rectangle_matches, find_rectangle_size_occurences, find_same_color_sub_rectangles,
    remove_moot_color_rectangles, split_single_match_and_multiple_matches,
    write_average_match_to_image, write_first_match_to_image, Rectangle,
};

fn main() {
    // logging.info("Loading pixelated image from %s" % pixelatedImagePath)
    println!("Loading pixelated image.");
    let pixelated_image = DynamicImage::from_decoder(
        PngDecoder::new(
            fs::File::open("F:\\Depix\\images\\testimages\\testimage2_pixels.png").unwrap(),
        )
        .unwrap(),
    )
    .unwrap();

    // logging.info("Loading search image from %s" % searchImagePath)
    println!("Loading search image.");
    let search_image = DynamicImage::from_decoder(
        PngDecoder::new(
            fs::File::open(
                "F:\\Depix\\images\\searchimages\\debruinseq_notepad_Windows10_close.png",
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut unpixelated_output_image = pixelated_image.clone();

    // logging.info("Finding color rectangles from pixelated space")
    println!("Finding color rectangles from pixelated space");
    let pixelated_rectangle = Rectangle {
        x: 0,
        y: 0,
        end_x: pixelated_image.width() - 1,
        end_y: pixelated_image.height() - 1,
    };
    let mut pixelated_sub_rectangles =
        find_same_color_sub_rectangles(&pixelated_image, pixelated_rectangle);
    // logging.info("Found %s same color rectangles" % len(pixelatedSubRectanges))
    println!(
        "Found {} same color rectangles",
        pixelated_sub_rectangles.len()
    );

    pixelated_sub_rectangles = remove_moot_color_rectangles(pixelated_sub_rectangles, None);
    // logging.info("%s rectangles left after moot filter" % len(pixelatedSubRectanges))
    println!(
        "{} rectangles left after moot filter",
        pixelated_sub_rectangles.len()
    );

    let rectangle_size_occurences =
        find_rectangle_size_occurences(pixelated_sub_rectangles.clone());
    // logging.info("Found %s different rectangle sizes" % len(rectangeSizeOccurences))
    println!(
        "Found {} different rectangle sizes",
        rectangle_size_occurences.len()
    );
    // if len(rectangeSizeOccurences) > max(10, pixelatedRectange.width * pixelatedRectange.height * 0.01):
    //      logging.warning("Too many variants on block size. Re-pixelating the image might help.")
    if rectangle_size_occurences.len()
        > max(
            10,
            (pixelated_rectangle.width() as f32 * pixelated_rectangle.height() as f32 * 0.01)
                as usize,
        )
    {
        println!("Warning: Too many variants on block size. Re-pixelation the image might help.");
    }

    // logging.info("Finding matches in search image")
    println!("Finding matches in search image");
    let rectangle_matches = find_rectangle_matches(
        rectangle_size_occurences,
        pixelated_sub_rectangles.clone(),
        &search_image,
        text_recovery::AverageType::GammaCorrected,
    );

    // logging.info("Removing blocks with no matches")
    println!("Removing blocks with no matches");
    pixelated_sub_rectangles =
        drop_empty_rectangle_matches(rectangle_matches.clone(), pixelated_sub_rectangles);

    // logging.info("Splitting single matches and multiple matches")
    println!("Splitting single matches and multiple matches");
    let (single_results, pixelated_sub_rectangles) = split_single_match_and_multiple_matches(
        pixelated_sub_rectangles,
        rectangle_matches.clone(),
    );

    // logging.info("[%s straight matches | %s multiple matches]" % (len(singleResults), len(pixelatedSubRectanges)))
    println!(
        "[{} straight matches | {} multiple matches]",
        single_results.len(),
        pixelated_sub_rectangles.len()
    );

    // logging.info("Trying geometrical matches on single-match squares")
    println!("Trying geometrical matches on single-match squares");
    let (single_results, pixelated_sub_rectangles) = find_geometric_matches_for_single_results(
        single_results,
        pixelated_sub_rectangles,
        rectangle_matches.clone(),
    );

    // logging.info("[%s straight matches | %s multiple matches]" % (len(singleResults), len(pixelatedSubRectanges)))
    println!(
        "[{} straight matches | {} multiple matches]",
        single_results.len(),
        pixelated_sub_rectangles.len()
    );

    // logging.info("Writing single match results to output")
    println!("Writing single match results to output");
    write_first_match_to_image(
        single_results,
        rectangle_matches.clone(),
        &search_image,
        &mut unpixelated_output_image,
    );
    write_average_match_to_image(
        pixelated_sub_rectangles,
        rectangle_matches,
        &mut unpixelated_output_image,
    );

    unpixelated_output_image
        .save("output.png")
        .expect("Failed whilst image creation");
}
