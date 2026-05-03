# import requests

# # Make a GET request to a public API (e.g., JSONPlaceholder)
# url = "https://sipapiaks.totalwine.com/purchase-order/pooptconfigapi/api/v1.0/stores"
# response = requests.get(url)

# # Check the HTTP status code
# print("Status Code:", response.status_code)

# # Check if request was successful (2xx status codes)
# if response.status_code == 200:
#     print("✅ Success!")
#     # Parse JSON response
#     data = response.json()
#     print("Response Data:", data)
# else:
#     print(f"❌ Failed with status code {response.status_code}")
#     # Optionally print error text
#     print("Error Response:", response.text)
    
    


# import subprocess

# def strong_compress(input_path, output_path):
#     subprocess.run([
#         "gs",
#         "-sDEVICE=pdfwrite",
#         "-dCompatibilityLevel=1.4",
#         "-dPDFSETTINGS=/ebook",
#         "-dNOPAUSE",
#         "-dQUIET",
#         "-dBATCH",
#         f"-sOutputFile={output_path}",
#         input_path
#     ])

# strong_compress("ilovepdf_merged.pdf", "Birth_certificate_compressed.pdf")



from PyPDF2 import PdfReader, PdfWriter

def compress_pdf(input_path, output_path):
    reader = PdfReader(input_path)
    writer = PdfWriter()

    for page in reader.pages:
        page.compress_content_streams()  # compress content
        writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)

compress_pdf("ilovepdf_merged.pdf", "birth_certificate_compress.pdf")