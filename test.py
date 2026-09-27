from pypdf import PdfReader

reader = PdfReader("./data/2608.20316v1_Pandora's AI Model Routing Box Efficient Allocation with Costly Value Estimation.pdf")

len(reader.pages)

page = reader.pages[0]
print(page.extract_text())