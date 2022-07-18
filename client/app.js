var writer = document.getElementById("output");
var modal = document.getElementById("myModal");
var span = document.getElementsByClassName("close")[0];

function get_Gender(){
    if ($("input[name='inlineRadioOptions']").prop("checked")){
        return 1
    }
    return 0;
}
function get_ResidenceType(){
    if ($("input[name='Residence']").prop("checked")){
        return 1
    }
    return 0;
}
function get_MarriageStatus(){
    if ($("input[name='EverMarried']").prop("checked")){
        return 1
    }
    return 0
}
function get_heartIssues(){
    if ($("input[name='heartIssues']").prop("checked")){
        return 1
    }
    return 0
}
function get_Hypertension(){
    if ($("input[name='Hypertension']").prop("checked")){
        return 1
    }
    return 0
}
span.onclick = function() {
    modal.style.display = "none";
  }
  
// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
if (event.target == modal) {
    modal.style.display = "none";
}
}
console.log('Hi')
function getlikelihood(){
    var age = $('#Age').val();
    var bmi = $('#bmi').val();
    var glucose = $('#glucose').val();
    var gender = get_Gender();
    var hypertension = get_Hypertension();
    var heartIssues = get_heartIssues();
    var workType = $("#worktype_dropdown").val();
    var smoking = $("#smoking_dropdown").val();
    var url = "http://127.0.0.1:5000/stroke_likelihood";
    var post_data = {
        age: parseInt(age),
        bmi: parseFloat(bmi),
        avg_glucose_level: parseFloat(glucose),
        gender: parseInt(gender),
        residenceType: parseInt(get_ResidenceType()),
        everMarried: parseInt(get_MarriageStatus()),
        hypertension: parseInt(hypertension),
        heartIssues: parseInt(heartIssues),
        workType: parseInt(workType),
        smoking: parseInt(smoking)
    };
    $.post(url, post_data, function(data, status){
        modal.style.display = "block";
        writer.innerHTML = "Likelihood of you having a stroke is " + data.likelihood.toString() + "%.";
    });
}
