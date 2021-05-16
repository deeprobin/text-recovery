use std::collections::{HashMap, HashSet};

use image::{DynamicImage, GenericImage, GenericImageView, Pixel, Rgba};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rectangle {
    pub x: u32,
    pub y: u32,
    pub end_x: u32,
    pub end_y: u32,
}

impl Rectangle {
    pub fn width(&self) -> u32 {
        self.end_x - self.x
    }
    pub fn height(&self) -> u32 {
        self.end_y - self.y
    }
}

#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColoredRectangle {
    pub rectangle: Rectangle,
    pub color: Rgba<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RectangleMatch {
    pub x: u32,
    pub y: u32,
    pub data: Vec<Rgba<u8>>,
}

pub fn find_same_color_rectangle(
    image: &DynamicImage,
    start_x: u32,
    start_y: u32,
    max_x: u32,
    max_y: u32,
) -> ColoredRectangle {
    let color = image.get_pixel(start_x, start_y).to_rgba();

    let mut width = 0u32;
    let mut height = 0u32;

    for x in start_x..max_x {
        let pixel = image.get_pixel(x, start_y);
        if pixel.to_rgba() == color {
            width += 1;
        } else {
            break;
        }
    }

    for y in start_y..max_y {
        let pixel = image.get_pixel(start_x, y);
        if pixel.to_rgba() == color {
            height += 1;
        } else {
            break;
        }
    }

    for test_x in start_x..(start_x + width) {
        for test_y in start_y..(start_y + height) {
            let pixel = image.get_pixel(test_x, test_y);
            if pixel.to_rgba() != color {
                return ColoredRectangle {
                    color,
                    rectangle: Rectangle {
                        x: start_x,
                        y: start_y,
                        end_x: test_x,
                        end_y: test_y,
                    },
                };
            }
        }
    }

    ColoredRectangle {
        color,
        rectangle: Rectangle {
            x: start_x,
            y: start_y,
            end_x: start_x + width,
            end_y: start_y + height,
        },
    }
}

pub fn find_same_color_sub_rectangles(
    image: &DynamicImage,
    rectangle: Rectangle,
) -> Vec<ColoredRectangle> {
    let mut same_color_rectangles: Vec<ColoredRectangle> = Vec::new();

    let mut x = rectangle.x;
    let max_x = rectangle.x + rectangle.end_x + 1;
    let max_y = rectangle.y + rectangle.end_y + 1;

    while x < max_x {
        let mut y = rectangle.y;
        let mut same_color_rectangle: Option<ColoredRectangle> = None;
        while y < max_y {
            let rect = find_same_color_rectangle(&image, x, y, max_x, max_y);
            same_color_rectangle = Some(rect);
            same_color_rectangles.push(rect);

            // logging.info("Found rectangle at (%s, %s) with size (%s,%s) and color %s" % (x, y, sameColorRectange.width,sameColorRectange.height,sameColorRectange.color))
            println!(
                "Found rectangle at ({}, {}) with size ({},{}) and color {:?}",
                rect.rectangle.x,
                rect.rectangle.y,
                rect.rectangle.width(),
                rect.rectangle.height(),
                rect.color.0
            );

            y += rect.rectangle.height();
        }
        if let Some(rect) = same_color_rectangle {
            x += rect.rectangle.width();
        }
    }
    same_color_rectangles
}

pub fn remove_moot_color_rectangles(
    vec: Vec<ColoredRectangle>,
    editor_background_color: Option<[u8; 3]>,
) -> Vec<ColoredRectangle> {
    let mut new_vec: Vec<ColoredRectangle> = Vec::new();

    for color_rectangle in vec {
        if color_rectangle.color[0] == 0
            && color_rectangle.color[1] == 0
            && color_rectangle.color[2] == 0
        {
            continue;
        }
        if color_rectangle.color[0] == 255
            && color_rectangle.color[1] == 255
            && color_rectangle.color[2] == 255
        {
            continue;
        }
        if let Some(editor_background_color) = editor_background_color {
            if color_rectangle.color[0] == editor_background_color[0]
                && color_rectangle.color[1] == editor_background_color[1]
                && color_rectangle.color[2] == editor_background_color[2]
            {
                continue;
            }
        }
        new_vec.push(color_rectangle);
    }

    new_vec
}

pub fn find_rectangle_size_occurences(vec: Vec<ColoredRectangle>) -> HashMap<(u32, u32), usize> {
    let mut map: HashMap<(u32, u32), usize> = HashMap::new();

    for color_rectangle in vec {
        let size = (
            color_rectangle.rectangle.width(),
            color_rectangle.rectangle.height(),
        );

        match map.get_mut(&size) {
            Some(s) => {
                *s += 1;
            }
            None => {
                map.insert(size, 1);
            }
        }
    }

    map
}

pub fn srgb2lin(s: f64) -> f64 {
    if s <= 0.0404482362771082 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

pub fn lin2srgb(lin: f64) -> f64 {
    if lin > 0.0031308 {
        1.055 * lin.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * lin
    }
}

pub enum AverageType {
    GammaCorrected,
    Linear,
}

pub fn find_rectangle_matches(
    rectangle_size_occurences: HashMap<(u32, u32), usize>,
    pixelated_sub_rectangles: Vec<ColoredRectangle>,
    search_image: &DynamicImage,
    average_type: AverageType,
) -> HashMap<(u32, u32), Vec<RectangleMatch>> {
    let mut rectangle_matches: HashMap<(u32, u32), Vec<RectangleMatch>> = HashMap::new();

    for rectangle_size_occurence in rectangle_size_occurences {
        let rectangle_size = rectangle_size_occurence.0;
        let rectangle_width = rectangle_size.0;
        let rectangle_height = rectangle_size.1;
        let pixels_in_rectangle = rectangle_width * rectangle_height;

        let mut matching_rectangles: Vec<ColoredRectangle> = Vec::new();
        for color_rectangle in &pixelated_sub_rectangles {
            if (
                color_rectangle.rectangle.width(),
                color_rectangle.rectangle.height(),
            ) == rectangle_size
            {
                matching_rectangles.push(*color_rectangle);
            }
        }

        // TODO: NEEDS PERFORMANCE OPTIMIZATION (Maybe rayon / paralellization?) (Maybe GPU computation?)
        for x in 0..(search_image.width() - rectangle_width) {
            for y in 0..(search_image.height() - rectangle_height) {
                let mut r = 0u32;
                let mut g = 0u32;
                let mut b = 0u32;

                let mut match_data: Vec<Rgba<u8>> = Vec::new();

                for xx in 0..rectangle_width {
                    for yy in 0..rectangle_height {
                        let new_pixel = search_image.get_pixel(x + xx, y + yy);
                        match_data.push(new_pixel);

                        let (rr, gg, bb) = match average_type {
                            AverageType::GammaCorrected => {
                                (new_pixel.0[0], new_pixel.0[1], new_pixel.0[2])
                            }
                            AverageType::Linear => {
                                let mut new_pixel_linear = new_pixel.0;
                                for (i, v) in new_pixel.0.iter().enumerate() {
                                    new_pixel_linear[i] = srgb2lin(*v as f64 / 255f64) as u8;
                                }
                                (
                                    new_pixel_linear[0],
                                    new_pixel_linear[1],
                                    new_pixel_linear[2],
                                )
                            }
                        };

                        r += rr as u32;
                        g += gg as u32;
                        b += bb as u32;
                    }
                }

                let average_color = match average_type {
                    AverageType::GammaCorrected => (
                        r as u32 / pixels_in_rectangle,
                        g as u32 / pixels_in_rectangle,
                        g as u32 / pixels_in_rectangle,
                    ),
                    AverageType::Linear => {
                        let new_r = (lin2srgb(r as f64 / pixels_in_rectangle as f64) as u32) * 255;
                        let new_g = (lin2srgb(g as f64 / pixels_in_rectangle as f64) as u32) * 255;
                        let new_b = (lin2srgb(b as f64 / pixels_in_rectangle as f64) as u32) * 255;

                        (new_r, new_g, new_b)
                    }
                };

                for matching_rectangle in &matching_rectangles {
                    if !rectangle_matches.iter().any(|m| {
                        m.0 .0 == matching_rectangle.rectangle.x
                            && m.0 .1 == matching_rectangle.rectangle.y
                    }) {
                        rectangle_matches.insert(
                            (
                                matching_rectangle.rectangle.x,
                                matching_rectangle.rectangle.y,
                            ),
                            Vec::new(),
                        );
                    }

                    if (
                        matching_rectangle.color[0] as u32,
                        matching_rectangle.color[1] as u32,
                        matching_rectangle.color[2] as u32,
                    ) == average_color
                    {
                        let new_rectangle_match = RectangleMatch {
                            x: matching_rectangle.rectangle.x,
                            y: matching_rectangle.rectangle.y,
                            data: match_data.clone(),
                        };
                        if let Some(m) = rectangle_matches.get_mut(&(
                            matching_rectangle.rectangle.x,
                            matching_rectangle.rectangle.y,
                        )) {
                            m.push(new_rectangle_match);
                        }
                    }
                }
            }
            if x % 64 == 0 {
                println!(
                    "Scanning in searchImage {}/{}",
                    x,
                    search_image.width() - rectangle_width
                );
            }
        }
    }
    rectangle_matches
}

pub fn drop_empty_rectangle_matches(
    rectangle_matches: HashMap<(u32, u32), Vec<RectangleMatch>>,
    pixelated_sub_rectangles: Vec<ColoredRectangle>,
) -> Vec<ColoredRectangle> {
    let mut new_pixelated_sub_rectangles: Vec<ColoredRectangle> = Vec::new();

    for pixelated_sub_rectangle in pixelated_sub_rectangles {
        if (*rectangle_matches
            .get(&(
                pixelated_sub_rectangle.rectangle.x,
                pixelated_sub_rectangle.rectangle.y,
            ))
            .unwrap())
        .len()
            > 0
        {
            new_pixelated_sub_rectangles.push(pixelated_sub_rectangle);
        }
    }

    new_pixelated_sub_rectangles
}

pub fn split_single_match_and_multiple_matches(
    pixelated_sub_rectangles: Vec<ColoredRectangle>,
    rectangle_matches: HashMap<(u32, u32), Vec<RectangleMatch>>,
) -> (Vec<ColoredRectangle>, Vec<ColoredRectangle>) {
    let mut new_pixelated_sub_rectangles: Vec<ColoredRectangle> = Vec::new();
    let mut single_results: Vec<ColoredRectangle> = Vec::new();

    for color_rectangle in pixelated_sub_rectangles {
        let first_match_data = &rectangle_matches
            .get(&(color_rectangle.rectangle.x, color_rectangle.rectangle.y))
            .unwrap()[0]
            .data;
        let mut single_match = true;

        for rectangle_match in rectangle_matches
            .get(&(color_rectangle.rectangle.x, color_rectangle.rectangle.y))
            .unwrap()
        {
            if first_match_data != &rectangle_match.data {
                single_match = false;
                break;
            }
        }

        if single_match {
            single_results.push(color_rectangle);
        } else {
            new_pixelated_sub_rectangles.push(color_rectangle);
        }
    }

    (single_results, new_pixelated_sub_rectangles)
}

#[inline]
pub fn is_neighbor(pixel_a: Rectangle, pixel_b: Rectangle) -> bool {
    let x_delta: i32 = pixel_a.x as i32 - pixel_b.x as i32;
    let y_delta: i32 = pixel_a.y as i32 - pixel_b.y as i32;

    let negated_pixel_a_width = -(pixel_a.width() as i32);
    let negated_pixel_a_height = -(pixel_a.height() as i32);

    let x_condition =
        x_delta == pixel_b.width() as i32 || x_delta == 0 || x_delta == negated_pixel_a_width;
    let y_condition =
        y_delta == pixel_b.height() as i32 || y_delta == 0 || y_delta == negated_pixel_a_height;

    x_condition && y_condition && pixel_a != pixel_b
}
/*
fn findGeometricMatchesForSingleResults<T0, T1, T2, RT>(
    single_results: Vec<ColoredRectangle>,
    pixelated_sub_rectangles: Vec<ColoredRectangle>,
    rectangle_matches: HashMap<(u32, u32), Vec<RectangleMatch>>,
) -> (Vec<ColoredRectangle>, Vec<ColoredRectangle>) {
    let mut newPixelatedSubRectanges = pixelated_sub_rectangles.clone();
    let mut newSingleResults = single_results.clone();
    let matchCount: HashMap<ColoredRectangle, i32> = HashMap::new();
    let dataSeen = HashSet::new();
    for singleResult in single_results {
        for pixelatedSubRectange in pixelated_sub_rectangles {
            if !is_neighbor(singleResult.rectangle, pixelatedSubRectange.rectangle) {
                continue;
            }
            if matchCount.iter().any(|(&x, y)| x == pixelatedSubRectange)
                && matchCount[&pixelatedSubRectange] > 1
            {
                break;
            }
            for singleResultMatch in &rectangle_matches[&(singleResult.rectangle.x, singleResult.rectangle.y)] {
                for compareMatch in
                &rectangle_matches[&(pixelatedSubRectange.rectangle.x, pixelatedSubRectange.rectangle.y)]
                {
                    let xDistance = (singleResult.rectangle.x - pixelatedSubRectange.rectangle.x);
                    let yDistance = (singleResult.rectangle.y - pixelatedSubRectange.rectangle.y);
                    let xDistanceMatches = (singleResultMatch.x - compareMatch.x);
                    let yDistanceMatches = (singleResultMatch.y - compareMatch.y);
                    if xDistance == xDistanceMatches && yDistance == yDistanceMatches {
                        if dataSeen
                            .iter()
                            .all(|&x| x != (compareMatch.data, singleResultMatch.data))
                        {
                            dataSeen.insert((compareMatch.data, singleResultMatch.data));
                            if matchCount.iter().all(|(&x, _)| x != pixelatedSubRectange) {
                                matchCount[&pixelatedSubRectange] = 1;
                            } else {
                                matchCount[&pixelatedSubRectange] += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    for pixelatedSubRectange in matchCount {
        if matchCount[&pixelatedSubRectange.0] == 1 {
            newSingleResults.push(pixelatedSubRectange.0);
            let item_pos = newPixelatedSubRectanges.iter().position(|x| *x == pixelatedSubRectange.0).unwrap();
            newPixelatedSubRectanges.remove(item_pos);
        }
    }
    return (newSingleResults, newPixelatedSubRectanges);
}
*/

pub fn find_geometric_matches_for_single_results(
    single_results: Vec<ColoredRectangle>,
    pixelated_sub_rectangles: Vec<ColoredRectangle>,
    rectangle_matches: HashMap<(u32, u32), Vec<RectangleMatch>>,
) -> (Vec<ColoredRectangle>, Vec<ColoredRectangle>) {
    let mut new_pixelated_sub_rectangles = pixelated_sub_rectangles.clone();
    let mut new_single_results = single_results.clone();
    let mut match_count: HashMap<ColoredRectangle, i32> = HashMap::new();
    let mut data_seen: HashSet<(&Vec<Rgba<u8>>, &Vec<Rgba<u8>>)> = HashSet::new();

    for single_result in single_results {
        for pixelated_sub_rectangle in &pixelated_sub_rectangles {
            if !is_neighbor(single_result.rectangle, pixelated_sub_rectangle.rectangle) {
                continue;
            }
            if match_count.contains_key(pixelated_sub_rectangle)
                && match_count.get(pixelated_sub_rectangle).unwrap() > &1
            {
                break;
            }

            for single_result_match in
                &rectangle_matches[&(single_result.rectangle.x, single_result.rectangle.y)]
            {
                for compare_match in &rectangle_matches[&(
                    pixelated_sub_rectangle.rectangle.x,
                    pixelated_sub_rectangle.rectangle.y,
                )] {
                    let x_distance =
                        single_result.rectangle.x - pixelated_sub_rectangle.rectangle.x;
                    let y_distance =
                        single_result.rectangle.y - pixelated_sub_rectangle.rectangle.y;
                    let x_distance_matches = single_result_match.x - compare_match.x;
                    let y_distance_matches = single_result_match.y - compare_match.y;

                    if x_distance == x_distance_matches && y_distance == y_distance_matches {
                        let repr = (&compare_match.data, &single_result_match.data);
                        if !data_seen.contains(&repr) {
                            data_seen.insert(repr);

                            if match_count.contains_key(&pixelated_sub_rectangle) {
                                if let Some(rect) = match_count.get_mut(&pixelated_sub_rectangle) {
                                    *rect += 1;
                                }
                            } else {
                                if let Some(rect) = match_count.get_mut(&pixelated_sub_rectangle) {
                                    *rect = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for pixelated_sub_rectangle in match_count {
        if pixelated_sub_rectangle.1 == 1 {
            new_single_results.push(pixelated_sub_rectangle.0);
            if let Some(pos) = new_pixelated_sub_rectangles
                .iter()
                .position(|x| *x == pixelated_sub_rectangle.0)
            {
                new_pixelated_sub_rectangles.remove(pos);
            }
        }
    }

    (new_single_results, new_pixelated_sub_rectangles)
}

pub fn write_first_match_to_image(
    single_match_rectangles: Vec<ColoredRectangle>,
    rectangle_matches: HashMap<(u32, u32), Vec<RectangleMatch>>,
    search_image: &DynamicImage,
    unpixelated_output_image: &mut DynamicImage,
) {
    for single_result in single_match_rectangles {
        let single_match =
            &rectangle_matches[&(single_result.rectangle.x, single_result.rectangle.y)][0];

        for x in 0..single_result.rectangle.width() {
            for y in 0..single_result.rectangle.height() {
                let origin_pixel = search_image.get_pixel(single_match.x + x, single_match.y + y);
                unpixelated_output_image.put_pixel(
                    single_result.rectangle.x + x,
                    single_result.rectangle.y + y,
                    origin_pixel,
                );
            }
        }
    }
}

pub fn write_average_match_to_image(
    pixelated_sub_rectangles: Vec<ColoredRectangle>,
    rectangle_matches: HashMap<(u32, u32), Vec<RectangleMatch>>,
    unpixelated_output_image: &mut DynamicImage,
) {
    for pixelated_sub_rectangle in pixelated_sub_rectangles {
        let coordinate = (
            pixelated_sub_rectangle.rectangle.x,
            pixelated_sub_rectangle.rectangle.y,
        );
        let matches = &rectangle_matches[&coordinate];

        let mut new_image = DynamicImage::new_rgb8(
            pixelated_sub_rectangle.rectangle.width(),
            pixelated_sub_rectangle.rectangle.height(),
        );

        for image_match in matches {
            let mut data_count = 0;
            for x in 0..pixelated_sub_rectangle.rectangle.width() {
                for y in 0..pixelated_sub_rectangle.rectangle.height() {
                    let pixel_data = image_match.data[data_count];
                    data_count += 1;

                    // By default white
                    let current_pixel_color: [u8; 4] = [255; 4];

                    let r = pixel_data[0] + current_pixel_color[0] / 2;
                    let g = pixel_data[1] + current_pixel_color[1] / 2;
                    let b = pixel_data[2] + current_pixel_color[2] / 2;

                    let average_pixel: [u8; 4] = [r, g, b, current_pixel_color[3]];
                    let rgba_pixel = Rgba::from(average_pixel);
                    new_image.put_pixel(x, y, rgba_pixel);
                }
            }
        }

        for x in 0..pixelated_sub_rectangle.rectangle.width() {
            for y in 0..pixelated_sub_rectangle.rectangle.height() {
                let current_pixel = new_image.get_pixel(x, y);
                unpixelated_output_image.put_pixel(
                    pixelated_sub_rectangle.rectangle.x,
                    pixelated_sub_rectangle.rectangle.y,
                    current_pixel,
                );
            }
        }
    }
}

/*
// PYTHON TODO
def findGeometricMatchesForSingleResults(singleResults, pixelatedSubRectanges, rectangleMatches):

    newPixelatedSubRectanges = pixelatedSubRectanges[:]
    newSingleResults = singleResults[:]
    matchCount = {}
    dataSeen = set()

    for singleResult in singleResults:
        for pixelatedSubRectange in pixelatedSubRectanges:
            if not isNeighbor(singleResult, pixelatedSubRectange):
                continue
            if pixelatedSubRectange in matchCount and matchCount[pixelatedSubRectange] > 1:
                break

            # use relative position to determine its neighbors
            for singleResultMatch in rectangleMatches[(singleResult.x, singleResult.y)]:
                for compareMatch in rectangleMatches[(pixelatedSubRectange.x, pixelatedSubRectange.y)]:

                    xDistance = singleResult.x - pixelatedSubRectange.x
                    yDistance = singleResult.y - pixelatedSubRectange.y
                    xDistanceMatches = singleResultMatch.x - compareMatch.x
                    yDistanceMatches = singleResultMatch.y - compareMatch.y

                    if xDistance == xDistanceMatches and yDistance == yDistanceMatches:
                        if repr((compareMatch.data, singleResultMatch.data)) not in dataSeen:

                            dataSeen.add(repr((compareMatch.data, singleResultMatch.data)))

                            if pixelatedSubRectange not in matchCount:
                                matchCount[pixelatedSubRectange] = 1
                            else:
                                matchCount[pixelatedSubRectange] += 1

    for pixelatedSubRectange in matchCount:
        if matchCount[pixelatedSubRectange] == 1:
            newSingleResults.append(pixelatedSubRectange)
            newPixelatedSubRectanges.remove(pixelatedSubRectange)

    return newSingleResults, newPixelatedSubRectanges


def writeFirstMatchToImage(singleMatchRectangles, rectangleMatches, searchImage, unpixelatedOutputImage):

    for singleResult in singleMatchRectangles:
        singleMatch = rectangleMatches[(singleResult.x,singleResult.y)][0]

        for x in range(singleResult.width):
            for y in range(singleResult.height):

                color = searchImage.imageData[singleMatch.x+x][singleMatch.y+y]
                unpixelatedOutputImage.putpixel((singleResult.x+x,singleResult.y+y), color)


def writeRandomMatchesToImage(pixelatedSubRectanges, rectangleMatches, searchImage, unpixelatedOutputImage):

    for singleResult in pixelatedSubRectanges:

        singleMatch = choice(rectangleMatches[(singleResult.x,singleResult.y)])

        for x in range(singleResult.width):
            for y in range(singleResult.height):

                color = searchImage.imageData[singleMatch.x+x][singleMatch.y+y]
                unpixelatedOutputImage.putpixel((singleResult.x+x,singleResult.y+y), color)


def writeAverageMatchToImage(pixelatedSubRectanges, rectangleMatches, searchImage, unpixelatedOutputImage):

    for pixelatedSubRectange in pixelatedSubRectanges:

        coordinate = (pixelatedSubRectange.x, pixelatedSubRectange.y)
        matches = rectangleMatches[coordinate]

        img = Image.new('RGB', (pixelatedSubRectange.width, pixelatedSubRectange.height), color = 'white')

        for match in matches:

            dataCount = 0
            for x in range(pixelatedSubRectange.width):
                for y in range(pixelatedSubRectange.height):

                    pixelData = match.data[dataCount]
                    dataCount += 1
                    currentPixel = img.getpixel((x,y))[0:3]

                    r = int((pixelData[0]+currentPixel[0])/2)
                    g = int((pixelData[1]+currentPixel[1])/2)
                    b = int((pixelData[2]+currentPixel[2])/2)

                    averagePixel = (r,g,b)

                    img.putpixel((x,y), averagePixel)

        for x in range(pixelatedSubRectange.width):
            for y in range(pixelatedSubRectange.height):

                currentPixel = img.getpixel((x,y))[0:3]
                unpixelatedOutputImage.putpixel((pixelatedSubRectange.x+x,pixelatedSubRectange.y+y), currentPixel)
*/
