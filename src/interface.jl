function diff_save_interface(root, fname = "")
    if fname !== ""
        return fname
    end
    l = length(splitpath(root))
    dir = pwd()
    cd(root)
    fname = save_dialog("Save Result", Null(), (GtkFileFilter("*.jld2", name="All supported formats"), "*.jld2"))
    cd(dir)
    if splitext(fname)[end] == ".jld2"
        return fname
    elseif splitext(fname)[end] == ""
        return fname*".jld2"
    else 
        error("invalid filename")
    end
end

function safe_save(fname, res)
    if !(fname !== "" && splitext(fname)[end] == ".jld2")
        error("invalid filename")
    end
    save(fname, res)
    println("Saving $fname")
end