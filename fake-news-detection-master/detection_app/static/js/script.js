var json_obj=[];
var json;
$(document).on('submit', '#fm', (e) => {
    $('#predict').attr('disabled', true);
    e.preventDefault();
    json = document.getElementById('news').value;
    $("#result").removeClass();
    console.log(json);
    document.getElementById("result").innerHTML = '';
    $('#loading').append(`  
    <div class="spinner-border" role="status">
    <span class="sr-only">Loading...</span>
    </div>
    `)
    $.ajax({
        type: 'POST',
        url: '',
        data: {
            csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val(),
            news : json
        },
        success: function (data) {
            // alert("success");
            console.log(data);
            var result = data.result;
            console.log(result);
            $('#result')[0].style.display="block";
        //   alert($("#table tr.selected td:first").html());
            if(result==0){
                $('#result').addClass('alert alert-success');
                $('#result')[0].textContent="The Following news is most probably REAL!";
            }
            else{
                $('#result').addClass('alert alert-danger');
                $('#result')[0].textContent="The Following news is most probably FAKE!";
            }
            $('#predict').attr('disabled', false);
            document.getElementById("loading").innerHTML='';
        }
    })
})

