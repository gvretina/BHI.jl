using BHI
using Documenter

DocMeta.setdocmeta!(BHI, :DocTestSetup, :(using BHI); recursive=true)

makedocs(;
    modules=[BHI],
    authors="Christos Mermigkas, Giorgos Vretinaris",
    sitename="BHI.jl",
    format=Documenter.HTML(;
        canonical="https://gvretinaris.github.io/BHI.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gvretinaris/BHI.jl",
    devbranch="main",
)
