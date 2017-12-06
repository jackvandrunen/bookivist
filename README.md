# Bookivist

Painlessly create high quality PDF scans of books with any camera.

![An example page](examples/example1.jpg?raw=true "An example page")

## Operation

Bookivist requires Python 3, OpenCV 3.x, NumPy, PIL (Pillow), and img2pdf:

```bash
$ pip install -r requirements.txt
```

A full CLI API is currently in development (check `bookivist.py -h` for more
info). Operation is simple:

```bash
$ bookivist.py 'book/*.jpg'
```

Where `'book/*.jpg'` is any glob for expansion by the program internally (be
sure to place it in single quotes so bash does not do the expansion). Photos
are sorted based on their EXIF timestamp (more flexible options will be
available soon), cropped and binarized, and then dumped into an output PDF file
`output.pdf`. From there, you can also use a tool like
[OCRmyPDF](https://github.com/jbarlow83/OCRmyPDF) to add a layer of searchable
text to the scan:

```bash
$ ocrmypdf output.pdf output_ocr.pdf
```

## Tips

When taking the photos, a high contrast between the background and the page is
good (I used a black whiteboard for my tests), as well as bright lighting. Try
to capture the entire page, so the program has an easier time cropping.
