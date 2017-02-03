$(function() {
  var submit_form = function(e) {
    $.getJSON($SCRIPT_ROOT + '/_cluster', {
      query: $('input[name="query"]').val(),
    }, function(data) {
      $('#clusters').text(data.result);
      console.log(data.result) // HERE IS THE ARRAY OF IDS IN CLUSTERS
      $('input[name=query]').focus().select();
    });
    return false;
  };
  $('a#calculate').bind('click', submit_form);
  $('input[type=text]').bind('keydown', function(e) {
    if (e.keyCode == 13) {
      submit_form(e);
    }
  });
  $('input[name=query]').focus();
});

