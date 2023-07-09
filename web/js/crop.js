const FLOWER_CLASSES = {
  0: "setosa",
  1: "versicolour",
  2: "virginica",
};

var model;
//load model:
$("document").ready(async function () {
  // const response = await fetch("https://nhandang123.netlify.app/.netlify/functions/getJson");
  // const jsonContent = await response.json();
  model = await tf.loadGraphModel(
    "https://nhandang123.netlify.app/models/model.json"
  );
  console.log(model);

  $("#output").empty();
  //predict:
  let imEL = document.getElementById("img");
  imEL.onload = function () {
    $("#output").empty();
    predict();
  };
});

//predict function:
async function predict() {
  // transfor to tensor
  let images = document.getElementById("img");
  let img = tf.browser
    .fromPixels(images)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .reverse(2)
    .div(255.0)
    .expandDims();
  //predict:
  let predictions = await model.predict(img).data();
  console.log(predictions);
  //show result:
  let top5 = Array.from(predictions)
    .map(function (p, i) {
      return {
        probability: p,
        className: FLOWER_CLASSES[i],
      };
    })
    .sort(function (a, b) {
      return b.probability - a.probability;
    })
    .slice(0, 1);
  $("#output").empty();
  top5.forEach(function (p) {
    $("#output").append(`<p>${p.className}: ${p.probability.toFixed(6)}</p>`);
  });
}
//upload image:
$("#uploadbtn").click(function () {
  $("#fileinput").trigger("click");
});

var croppie = null;

// $("#fileinput").change(async function () {
document.getElementById("fileinput").addEventListener("change", function (e) {
  let file = this.files[0];
  let reader = new FileReader();
  reader.onload = function (event) {
    let dataURL = reader.result;
    $("#img").attr("src", dataURL);
    var croppieContainer = document.getElementById("img");
    // var resultContainer = document.getElementById("result");
    // Clear previous Croppie instance
    if (croppie !== null) {
      croppie.destroy();
      //   resultContainer.innerHTML = "";
    }
    var containerWidth = croppieContainer.offsetWidth;
    // var viewportSize = { width: containerWidth, height: containerWidth };
    var viewportSize = { width: "50%", height: "auto" };
    var boundarySize = { width: 100, height: 100 };

    // Initialize Croppie instance
    croppie = new Croppie(croppieContainer, {
      viewport: viewportSize,
      boundary: boundarySize,
    });

    // Bind image to Croppie
    croppie.bind({
      url: event.target.result,
      orientation: 1, // Set image orientation (optional)
    });
  };
  //   let file = $("#fileinput").prop("files")[0];
  reader.readAsDataURL(file);
});

const btn = document.querySelector(".movedown_btn");
btn.addEventListener("click", function () {
  window.scrollTo({
    top: document.body.scrollHeight,
    behavior: "smooth",
    duration: 2000,
  });
});
