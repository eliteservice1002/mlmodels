from flask import Flask, flash, request, redirect, url_for, render_template
# from flask_ngrok import run_with_ngrok
import os
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# weights = torch.load('large_ft.pkl')
weights = torch.load('medium_ft.pkl')

# cfg = GPT2Config(n_embd=1024,n_layer=24,n_head=16)
cfg = GPT2Config.from_json_file('DialoGPT/configs/345M/config.json')
model = GPT2LMHeadModel(cfg)

# fix misused key value
weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
weights.pop("lm_head.decoder.weight",None)

model.load_state_dict(weights)
model.to('cuda')
model.eval()

conditioned_tokens = []
generated_tokens = []
past = None
fst = True
DELIMITER = "/n"

def reinput(user_msg):
	global conditioned_tokens; global fst
	os.system('cls' if os.name == 'nt' else 'clear')
	  
	conditioned_tokens = tokenizer.decode(conditioned_tokens)

	print("Me: " + user_msg + "\n" + "Bot: ",end='')
	
	if fst:
	  fst = False
	  user_msg = "<|endoftext|>" + user_msg
  
	user_msg = "" + user_msg
	conditioned_tokens += user_msg 
	conditioned_tokens = tokenizer.encode(conditioned_tokens) 
	conditioned_tokens += [50256] # Append operator to prepend conversation history
  


def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
  """
  Credit: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
  """
  assert logits.dim() == 1  # batch size 1 for single word generation
  sorted_logits, sorted_indices = torch.sort(logits, descending=True)
  cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
  # remove tokens with cumulative probability above the threshold
  sorted_indices_to_remove = cumulative_probs > top_p
  # shift the indices to the right to keep also the first token above the threshold
  sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
  sorted_indices_to_remove[..., 0] = 0
  indices_to_remove = sorted_indices[sorted_indices_to_remove]
  logits[indices_to_remove] = filter_value
  return logits


def recalc(p=None):
	global conditioned_tokens
	global generated_tokens
	global past
  
  # for segment display purpose, keep 2 sets of tokens
	indexed_tokens = conditioned_tokens + generated_tokens
  
	tokens_tensor = torch.tensor([indexed_tokens])
  
	tokens_tensor = tokens_tensor.to('cuda')
	with torch.no_grad():
	    outputs,past = model(tokens_tensor, past=p)
	    predictions = outputs
	logits = predictions[0, -1, :]
	filtered_logits = top_p_filtering(logits)
	probabilities = F.softmax(filtered_logits, dim=-1)
	next_token = torch.multinomial(probabilities, 1)
	generated_tokens.append(next_token.item())
	return next_token.item()

def generate():
	global conditioned_tokens
	global generated_tokens
	global past
 
	while True:
		if len(tokenizer.decode(conditioned_tokens)) > 320:
		  dc = tokenizer.decode(conditioned_tokens)
		  dc = dc[len(dc)-320:]
		  idx = dc.find(DELIMITER)
		  if idx != -1:
		    dc = dc[idx+len(DELIMITER):]
		    conditioned_tokens = tokenizer.encode(dc)
    
		result = recalc()

		if result == 50256:
      
      # end-of-text : 50256
      # use this special token to split segments

			decoded_reply = tokenizer.decode(generated_tokens)
      
			to_print = decoded_reply
			if to_print.endswith("<|endoftext|>"):
			  to_print = to_print[:-len("<|endoftext|>")]
			  decoded_reply = to_print
			print(to_print)
			
			decoded_reply = "" + decoded_reply
			decoded_reply = decoded_reply+DELIMITER
			conditioned_tokens += (tokenizer.encode(decoded_reply))
    
			# Remove end of text tokens from feeding back into model
      # -- see here for reason https://github.com/huggingface/transformers/issues/429#issuecomment-479380117
			cond_str = tokenizer.decode(conditioned_tokens)
			cond_str = cond_str.replace("<|endoftext|>",DELIMITER) # Since "<end-of-text> (50256)" is mapped to the newline character (\n) by default when calling decode():
			conditioned_tokens = tokenizer.encode(cond_str)
    
# Uncomment to debug (print) conversation history that is re-fed to the model
# 			cond_str = tokenizer.decode(conditioned_tokens)
# 			cond_str = cond_str.replace("<|endoftext_|>","\n").replace(DELIMITER,"\n") # Since "<end-of-text> (50256)" is mapped to the newline character (\n) by default when calling decode():
# 			print("(condstr_dbg) " + cond_str)
# 			print(r"(/constr_dbg)")
    
			generated_tokens = []
			return to_print
			break



app = Flask(__name__)
# run_with_ngrok(app)
home_page = '''
<!DOCTYPE html>
<html>
  <title>Bot</title>
  <head>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
    <style>
      body {
        font-family: monospace;
      }
      h1 {
        background-color: yellow;
        display: inline-block;
        font-size: 3em;
        margin: 0;
        padding: 14px;
      }
      h3 {
        color: black;
        font-size: 20px;
        margin-top: 3px;
        text-align: center;
      }
      #chatbox {
        margin-left: auto;
        margin-right: auto;
        width: 40%;
        margin-top: 60px;
      }
      #userInput {
        margin-left: auto;
        margin-right: auto;
        width: 40%;
        margin-top: 60px;
      }
      #textInput {
        width: 90%;
        border: none;
        border-bottom: 3px solid black;
        font-family: monospace;
        font-size: 17px;
      }
      .userText {
        color: white;
        font-family: monospace;
        font-size: 17px;
        text-align: right;
        line-height: 30px;
      }
      .userText span {
        background-color: #808080;
        padding: 10px;
        border-radius: 2px;
      }
      .botText {
        color: white;
        font-family: monospace;
        font-size: 17px;
        text-align: left;
        line-height: 30px;
      }
      .botText span {
        background-color: #4169e1;
        padding: 10px;
        border-radius: 2px;
      }
      #tidbit {
        position: absolute;
        bottom: 0;
        right: 0;
        width: 300px;
      }
      .boxed {
        margin-left: auto;
        margin-right: auto;
        width: 78%;
        margin-top: 60px;
        border: 1px solid green;
      }
      .box {
        border: 2px solid black;
      }
    </style>
  </head>
  <body>
    <img />
    <center>
      <h1>
        <!-- <img
          src="https://user-images.githubusercontent.com/20112458/49326597-773b7280-f57a-11e8-853d-20ed61d18b0d.png"
          alt="CANDICE"
          style="width:40px;height:40px;"
        />Your Personal ChatBot -->
      </h1>
    </center>
    <h3>
      
    </h3>
    
    <div class="box"></div>
    <div class="boxed">
      <div>
        <div id="chatbox">
        
          <p class="botText">
            <span>Hi! Ask Questions</span>
          </p>
        </div>
        <div id="userInput">
          <input id="textInput" type="text" name="msg" placeholder="Message" />
        </div>
      </div>
      <script>
        function getBotResponse() {
          var rawText = $("#textInput").val();
          var userHtml = '<p class="userText"><span>' + rawText + "</span></p>";
          $("#textInput").val("");
          $("#chatbox").append(userHtml);
          document
            .getElementById("userInput")
            .scrollIntoView({ block: "start", behavior: "smooth" });
          $.get("/get", { msg: rawText }).done(function(data) {
            var botHtml = '<p class="botText"><span>' + data + "</span></p>";
            $("#chatbox").append(botHtml);
            document
              .getElementById("userInput")
              .scrollIntoView({ block: "start", behavior: "smooth" });
          });
        }
        $("#textInput").keypress(function(e) {
          if (e.which == 13) {
            getBotResponse();
          }
        });
        $(function() {
    $('#upload-file-btn').click(function() {
        var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/upload_file',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log('Success!');
            },
        });
    });
});
      </script>
    </div>
  </body>
</html>
'''

@app.route("/")
def home():
  return home_page
    # return render_template("home.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    reinput(userText)
    output = generate()
    return output

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=9999, debug=True)
# app.run()