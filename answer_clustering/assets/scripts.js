
function addCollapsible() {
    var coll = document.getElementsByClassName("collapsible");
    var i;
    for (i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function () {
            this.classList.toggle("active");
            var content = this.nextElementSibling;
            if (content.style.maxHeight) {
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
                // the timeout unsets the maxHeight parameter that is used to make a sliding animation
                // Necessary because if the content expands (e.g. data changes), the maxHeight will block auto expansion of the content div's height
                // the number in setTimeout should ideally match the animation time set in assets/collapsible.css (.collapsible-content `transition` property)
                setTimeout(function () {
                    content.style.maxHeight = "unset";
                }, 200)
            }
        });
    }
}

if (!window.dash_clientside) { window.dash_clientside = {}; }

window.dash_clientside.clientside = {

    addCollapsible: function (id) {
        addCollapsible();
        return window.dash_clientside.no_update
    },
}