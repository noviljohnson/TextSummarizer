import streamlit_authenticator as stauth
import streamlit as st

# name, authentication_status, username = authenticator.login('Login', 'main')
# if authentication_status:
#     authenticator.logout('Logout', 'main')
#     if username == 'jsmith':
#         st.write(f'Welcome *{name}*')
#         st.title('Application 1')
#     elif username == 'rbriggs':
#         st.write(f'Welcome *{name}*')
#         st.title('Application 2')
# elif authentication_status == False:
#     st.error('Username/password is incorrect')
# elif authentication_status == None:
#     st.warning('Please enter your username and password')


ui_url = "http://localhost:8501"

def main():
	"""Simple Login App"""

	st.title("Simple Login App")

	menu = ["Home","Login","SignUp"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")

	elif choice == "Login":
		st.subheader("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		# if st.sidebar.checkbox("Login"):
		# 	# if password == '12345':
		# 	create_usertable()
		# 	hashed_pswd = make_hashes(password)

		# 	result = login_user(username,check_hashes(password,hashed_pswd))
		# 	if result:

		# 		st.success("Logged In as {}".format(username))

		# 		task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
		# 		if task == "Add Post":
		# 			st.subheader("Add Your Post")

		# 		elif task == "Analytics":
		# 			st.subheader("Analytics")
		# 		elif task == "Profiles":
		# 			st.subheader("User Profiles")
		# 			user_result = view_all_users()
		# 			clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
		# 			st.dataframe(clean_db)
		# 	else:
		# 		st.warning("Incorrect Username/Password")





	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Email")
		new_password = st.text_input("Password",type='password')
		full_name = st.text_input('Full Name')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")



if __name__ == '__main__':
	main()